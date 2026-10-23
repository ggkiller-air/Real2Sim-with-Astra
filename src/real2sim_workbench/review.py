"""Reproducible multimodal Astra review with bounded, measured pose proposals."""
import json
import os
import shutil
import subprocess
from pathlib import Path
from .store import atomic_json

SCHEMA={'type':'object','properties':{
    'assessment':{'type':'string'},'visual_match':{'type':'string','enum':['good','partial','poor']},
    'limitations':{'type':'array','items':{'type':'string'}},
    'pose_search_ids':{'type':'array','items':{'type':'string'}},
    'adjustments':{'type':'array','items':{'type':'object','properties':{
        'id':{'type':'string'},'dx_m':{'type':'number'},'dy_m':{'type':'number'},'yaw_deg':{'type':'number'},'reason':{'type':'string'}},
        'required':['id','dx_m','dy_m','yaw_deg','reason'],'additionalProperties':False}}},
    'required':['assessment','visual_match','limitations','adjustments','pose_search_ids'],'additionalProperties':False}


def astra_review(scene, project, views, output):
    project,views,output=Path(project),Path(views),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    command=shutil.which(os.environ.get('R2S_CODEX_BIN','codex'))
    if not command:raise ValueError('本机未找到 Codex；多视角已保存，Astra 检查未运行')
    prompt_path=Path(__file__).parents[2]/'prompts/astra_scene_review.txt'
    prompt=prompt_path.read_text()+'\nScene context:\n'+json.dumps({
        'objects':[{'id':o.id,'name':o.name,'bbox':o.bbox,'size':o.size,'position':o.position,'yaw':o.yaw} for o in scene.objects if o.selected],
        'camera':scene.preprocessing.get('camera'),'physics':{k:v for k,v in scene.reports.get('validation',{}).items() if k!='initial_contacts'}},ensure_ascii=False)
    (output/'prompt.txt').write_text(prompt)
    atomic_json(output/'schema.json',SCHEMA)
    args=[command,'exec','--model','gpt-6-astra','--sandbox','read-only','--ephemeral',
          '--ignore-user-config','--skip-git-repo-check','-c','model_reasoning_effort="high"',
          '--output-schema',str(output/'schema.json'),'--output-last-message',str(output/'answer.json')]
    for path in (project/scene.image,views/'source_view.png',views/'views.png'):
        if path.is_file():args.extend(['--image',str(path.resolve())])
    args.append('-')
    with (output/'astra.log').open('w') as log:
        result=subprocess.run(args,input=prompt,text=True,stdout=log,stderr=subprocess.STDOUT,cwd=output,timeout=600)
    if result.returncode:raise RuntimeError('Astra 调用失败，日志保存在 '+str(output/'astra.log'))
    review=json.loads((output/'answer.json').read_text())
    # Do not treat model JSON as executable code or unrestricted scene state.
    ids={o.id for o in scene.objects if o.selected and not o.locked}
    pose_ids=review.get('pose_search_ids',[])
    if len(pose_ids)>2 or len(set(pose_ids))!=len(pose_ids) or any(oid not in ids for oid in pose_ids):
        raise ValueError('Astra 稳定姿态搜索目标无效')
    edits=review.get('adjustments',[])
    import math
    if len(edits)>3:raise ValueError('Astra 建议数量超出限制')
    seen=set()
    for edit in edits:
        if edit['id'] not in ids or edit['id'] in seen:raise ValueError('Astra 返回了无效或重复物体 ID')
        seen.add(edit['id'])
        if not all(math.isfinite(edit[k]) and abs(edit[k])<=bound for k,bound in [('dx_m',.1),('dy_m',.1),('yaw_deg',30)]):
            raise ValueError('Astra 位姿建议超出保守边界')
    return review


def review_and_refine(scene, project, output):
    """Astra proposes; native image measurements and physics accept or reject."""
    from . import physics
    import math
    project,output=Path(project),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    before_xml=physics.export_scene(scene,project,output/'before')
    physics.capture_views(scene,before_xml,output/'before_views',reference=project/scene.image)
    baseline=physics.silhouette_metrics(scene,project,before_xml)
    original_physics=physics.validate_scene(scene,before_xml,output/'before')
    review=astra_review(scene,project,output/'before_views',output/'astra')
    pose_report=None
    if review.get('pose_search_ids'):
        from .pose_search import search_stable_poses
        pose_report=search_stable_poses(scene,project,output/'pose_search',review['pose_search_ids'])
    candidate=scene.model_copy(deep=True)
    items={o.id:o for o in candidate.objects}
    for edit in review['adjustments']:
        obj=items[edit['id']];obj.position[0]+=edit['dx_m'];obj.position[1]+=edit['dy_m'];obj.yaw+=math.radians(edit['yaw_deg'])
    pose_accepted=bool(pose_report and any(item['accepted'] for item in pose_report['objects']))
    accepted=pose_accepted;reason='采纳稳定姿态候选并通过图像与物理复检' if pose_accepted else '未提出可采纳的有界位姿修改';after=pose_report['final_silhouette'] if pose_report else baseline;candidate_physics=None
    layout_accepted=False
    pose_baseline=after
    if review['adjustments']:
        try:
            physics.settle(candidate,project,output/'proposal_settle')
            xml=physics.export_scene(candidate,project,output/'proposal')
            candidate_physics=physics.validate_scene(candidate,xml,output/'proposal')
            after=physics.silhouette_metrics(candidate,project,xml)
            layout_accepted=bool(candidate_physics['physics_check_pass'] and after.get('mean_iou',0)>pose_baseline.get('mean_iou',0)+.001)
            reason='采纳平面位置修正并通过图像与物理复检' if layout_accepted else '平面位置建议未改善轮廓或未通过物理检查'
        except (ValueError,RuntimeError) as exc:
            reason='Rejected: '+str(exc)
    if layout_accepted:scene.objects=candidate.objects
    accepted=pose_accepted or layout_accepted
    if not layout_accepted:after=pose_baseline
    final_xml=physics.export_scene(scene,project,output/'simulation')
    validation=physics.validate_scene(scene,final_xml,output/'simulation')
    views=physics.capture_views(scene,final_xml,output/'views',reference=project/scene.image)
    report={'model':'gpt-6-astra','review':review,'accepted':accepted,'pose_accepted':pose_accepted,'layout_accepted':layout_accepted,'reason':reason,'before':baseline,'proposal':after,
            'pose_search':pose_report,'proposal_physics':candidate_physics,'original_physics_pass':original_physics['physics_check_pass'],
            'path':str(output.relative_to(project)), 'automated_visual_judgment_not_ground_truth':True}
    atomic_json(output/'review.json',report)
    scene.reports['astra_review']=report
    scene.reports['validation']=validation
    scene.reports['export']={'path':str((output/'simulation').relative_to(project)),'source_revision':scene.revision}
    scene.reports['views']={**views,'path':str((output/'views').relative_to(project))}
    atomic_json(output/'simulation'/'scene_record.json',scene.model_dump())
    shutil.copytree(output/'views',output/'simulation'/'evidence'/'views',dirs_exist_ok=True)
    shutil.copy2(output/'review.json',output/'simulation'/'evidence'/'astra_review.json')
    return report
