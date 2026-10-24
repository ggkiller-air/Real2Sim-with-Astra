from pathlib import Path
import numpy as np
import trimesh


def import_mesh(source, destination):
    """Normalize an imported rigid mesh to a unit AABB centered at the origin."""
    loaded = trimesh.load(source, force="scene", process=False)
    mesh = loaded.to_mesh()
    if len(mesh.vertices) > 500_000 or len(mesh.faces) > 1_000_000:
        raise ValueError("网格过大，请先简化到 50 万顶点以内")
    if len(mesh.vertices) < 4 or not np.isfinite(mesh.vertices).all() or np.any(mesh.extents < 1e-7):
        raise ValueError("需要具有三维体积的有效网格")
    original_size = mesh.extents.copy()
    mesh.vertices = (mesh.vertices - mesh.bounds.mean(axis=0)) / original_size
    # Write geometry only. Do not preserve external material/texture file references.
    normalized = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    normalized.export(destination, file_type="obj")
    return original_size.tolist()


def half_extent(obj):
    from scipy.spatial.transform import Rotation
    q=np.asarray(obj.orientation)
    rotation=Rotation.from_euler('z',obj.yaw)*Rotation.from_quat(q[[1,2,3,0]])
    return np.abs(rotation.as_matrix())@(np.array(obj.size)/2)


def homography(table, image_size):
    w, h = image_size
    corners = np.array(table.corners or [[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    # Reject folded or degenerate calibration polygons.
    edges = np.roll(corners, -1, axis=0) - corners
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not (np.all(cross > 1e-5) or np.all(cross < -1e-5)):
        raise ValueError("桌面四点需沿边界依次选择，不能交叉或共线")
    target = [[-table.width / 2, table.depth / 2], [table.width / 2, table.depth / 2],
              [table.width / 2, -table.depth / 2], [-table.width / 2, -table.depth / 2]]
    target=np.asarray(target)+np.asarray(table.center_xy)
    a, b = [], []
    for (x, y), (u, v) in zip(corners, target):
        a.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        b.extend([u, v])
    try:
        return np.append(np.linalg.solve(a, b), 1).reshape(3, 3)
    except np.linalg.LinAlgError as exc:
        raise ValueError("桌面标定退化，请重选四点") from exc


def support_order(scene):
    selected = {o.id: o for o in scene.objects if o.selected}
    order, done = [], {"table"}
    while len(order) < len(selected):
        ready = [o for o in selected.values() if o.id not in done and o.support in done]
        if not ready:
            raise ValueError("支撑物缺失或关系成环")
        for obj in ready:
            order.append(obj)
            done.add(obj.id)
    return order


def initialize(scene):
    matrix = homography(scene.table, scene.image_size)
    items = {o.id: o for o in scene.objects}
    for obj in support_order(scene):
        if obj.locked:
            continue
        x1, y1, x2, y2 = obj.bbox
        p = matrix @ [(x1+x2)/2, y2, 1]
        if abs(p[2]) < 1e-8:
            raise ValueError("物体落在标定平面的消失线上")
        xy = p[:2] / p[2]
        z = obj.size[2]/2
        if obj.support != "table":
            parent = items[obj.support]
            xy = np.array(parent.position[:2])
            z += parent.position[2] + parent.size[2]/2
        obj.position = [float(xy[0]), float(xy[1]), float(z + .0002)]
    return {"method": "image_bottom_anchor_to_plane", "metric_calibration": scene.table.calibration_source,
            "limitations": ["使用可见框底部作为接触点近似，未估计相机或深度", "父子支撑使用包围盒顶面，不能表示容器内腔或倚靠"]}


def optimize(scene):
    """Conservative, bounded XY search; does not alter mass/friction/geometry."""
    from scipy.optimize import minimize
    objs = [o for o in support_order(scene) if o.geometry != "none"]
    movable = [o for o in objs if not o.locked]
    if not movable:
        return {"method": "aabb_xy_search", "changed": False}
    anchors = np.array([o.position[:2] for o in movable])
    ext = {o.id: half_extent(o) for o in objs}
    items = {o.id: o for o in objs}
    fixed = {o.id: np.array(o.position[:2]) for o in objs}

    def objective(values):
        xy = dict(fixed)
        xy.update({o.id: v for o, v in zip(movable, values.reshape(-1, 2))})
        loss = float(np.square(values.reshape(-1, 2) - anchors).sum()) * .2
        for i, a in enumerate(objs):
            if a.support == "table":
                overflow = np.maximum(np.abs(xy[a.id]-np.asarray(scene.table.center_xy)) + ext[a.id][:2] - [scene.table.width/2, scene.table.depth/2], 0)
                loss += float(overflow @ overflow) * 200
            elif a.support in items:
                parent = items[a.support]
                overflow = np.maximum(np.abs(xy[a.id]-xy[parent.id]) + ext[a.id][:2] - ext[parent.id][:2], 0)
                loss += float(overflow @ overflow) * 200
            for b in objs[i+1:]:
                if abs(a.position[2]-b.position[2]) >= ext[a.id][2]+ext[b.id][2]-.0001:
                    continue
                overlap = ext[a.id][:2]+ext[b.id][:2] - np.abs(xy[a.id]-xy[b.id])
                if min(overlap) > 0:
                    loss += float(min(overlap)**2) * 200
        return loss

    before = objective(anchors.ravel())
    bounds = [(v-.15, v+.15) for v in anchors.ravel()]
    solved = minimize(objective, anchors.ravel(), method="Powell", bounds=bounds,
                      options={"maxiter": 80, "maxfev": 12000, "xtol": 1e-5, "ftol": 1e-6})
    improved = bool(np.isfinite(solved.fun) and solved.fun < before)
    if improved:
        for obj, xy in zip(movable, solved.x.reshape(-1, 2)):
            obj.position[:2] = [float(xy[0]), float(xy[1])]
    return {"method": "conservative_aabb_xy_search", "changed": improved, "objective_before": before,
            "objective_after": float(solved.fun) if improved else before, "solver_converged": bool(solved.success),
            "evaluations": int(solved.nfev), "max_translation_bound_m": .15,
            "limitations": ["仅优化 XY，保持高度与朝向", "保守包围盒代替精确碰撞距离；不是 REST3D 复现", "仍需独立 MuJoCo 检查与原图对照"]}


def prepare_visual_collision(mesh, destination, max_faces=14000):
    """Retain appearance separately from a bounded convex collision approximation.

    Input is oriented in world axes. Outputs use unit AABB coordinates so the
    existing editable SI dimensions remain the single source of scale.
    """
    from PIL import Image
    from scipy.spatial import cKDTree
    import coacd
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_vertices = np.asarray(mesh.vertices).copy()
    visual = mesh.visual.to_color() if hasattr(mesh.visual,"to_color") else mesh.visual
    colors = np.asarray(visual.vertex_colors)[:, :3]
    if len(mesh.faces) > max_faces:
        simple = mesh.simplify_quadric_decimation(face_count=max_faces)
        _, nearest = cKDTree(original_vertices).query(simple.vertices)
        colors = colors[nearest]
        mesh = simple
    size = mesh.extents.copy()
    center = mesh.bounds.mean(axis=0)
    if not np.isfinite(mesh.vertices).all() or np.any(size < 1e-6):
        raise ValueError("生成网格为空或退化")
    normalized = (mesh.vertices-center)/size
    # A triangle occupies its own padded tile, avoiding UV seam bleeding.
    count = len(mesh.faces)
    tile = 8
    side = int(np.ceil(np.sqrt(count)))
    atlas = np.zeros((side*tile, side*tile, 3), dtype=np.uint8)
    uv = np.empty((count*3, 2))
    xx,yy=np.meshgrid(np.arange(tile),np.arange(tile))
    u=np.clip((xx-1)/(tile-3),0,1);v=np.clip((yy-1)/(tile-3),0,1)
    barycentric=np.maximum(np.stack([1-u-v,u,v],axis=-1),0)
    barycentric/=barycentric.sum(axis=-1,keepdims=True)
    for i, face in enumerate(mesh.faces):
        tx, ty = (i % side)*tile, (i // side)*tile
        atlas[ty:ty+tile,tx:tx+tile] = (barycentric @ colors[face]).astype(np.uint8)
        uv[3*i:3*i+3] = [(np.array([tx+x+.5, side*tile-ty-y-.5])/(side*tile))
                           for x,y in ((1,1),(tile-2,1),(1,tile-2))]
    texture = destination.with_suffix('.png')
    Image.fromarray(atlas).save(texture)
    with destination.open('w') as out:
        for face in mesh.faces:
            for vid in face:
                v = normalized[vid]; c = colors[vid]/255
                out.write('v '+' '.join(f'{x:.9g}' for x in [*v,*c])+'\n')
        for u,v in uv:
            out.write(f'vt {u:.9g} {v:.9g}\n')
        for i in range(count):
            ids = [3*i+j+1 for j in range(3)]
            out.write('f '+' '.join(f'{j}/{j}' for j in ids)+'\n')
    # Decompose at uniform scale; nonuniform normalization is applied afterward.
    coacd.set_log_level('warn')
    uniform = max(size)
    parts = coacd.run_coacd(coacd.Mesh((mesh.vertices-center)/uniform, mesh.faces.astype(np.int32)),
                           threshold=.06, max_convex_hull=12, resolution=1500,
                           mcts_nodes=12, mcts_iterations=60, seed=42)
    if not parts:
        raise ValueError('碰撞分解未返回有效凸体')
    collision = []
    for i,(vertices,faces) in enumerate(parts):
        path = destination.with_name(destination.stem+f'_collision_{i:02d}.obj')
        trimesh.Trimesh(np.asarray(vertices)*uniform/size, faces, process=False).export(path)
        collision.append(path)
    return {'size':size.tolist(), 'center':center.tolist(), 'texture':texture,
            'collision':collision, 'faces':count, 'collision_method':'CoACD convex decomposition',
            'collision_parts':len(parts), 'appearance':'generated vertex colors baked into triangle atlas'}
