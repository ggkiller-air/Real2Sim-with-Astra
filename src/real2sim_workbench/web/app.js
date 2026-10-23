import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { OBJLoader } from "./vendor/OBJLoader.js";

const $ = (id) => document.getElementById(id);
const escape = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let scene = null,
  selected = null,
  mode = "select",
  dirty = false,
  busy = false,
  box = null,
  points = [],
  tablePoints = [],
  start = null;
let reference = null,
  masks = new Map(),
  modelStatus = {},
  previewGeneration = 0;
const colors = [
  "#d9754e",
  "#3d9b81",
  "#608bd2",
  "#b191cf",
  "#c4a141",
  "#cc6e8e",
  "#66a5b2",
];
const geometryLabels = {
  none: "无几何",
  proxy: "占位盒",
  imported: "导入网格",
  sam3d: "SAM 3D",
};
const annotationLabels = {
  rigid: "普通刚体",
  hinge: "铰链（仅标注）",
  drawer: "抽屉（仅标注）",
  cloth: "布料（仅标注）",
  soft: "软体（仅标注）",
};
const notice = (message, type = "") => {
  $("notice").textContent = message;
  $("notice").className = type;
};
async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch {
      body = { detail: response.statusText };
    }
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  }
  return response.json();
}
const json = (method, value) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(value),
});
function change() {
  dirty = true;
  $("saveState").textContent = "有未保存修改";
  renderReport();
}
const current = () => scene?.objects.find((o) => o.id === selected);
const fileURL = (path) => `/api/projects/${scene.id}/files/${path}`;

// Local three.js preview uses the same Z-up rigid dimensions and poses as export.
const world = new THREE.Scene();
world.background = new THREE.Color("#eaf0e9");
const camera = new THREE.PerspectiveCamera(43, 1, 0.005, 100);
camera.up.set(0, 0, 1);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  $("viewport").appendChild(renderer.domElement);
} catch (error) {
  notice(`三维预览不可用：${error.message}`, "error");
}
const controls = renderer
  ? new OrbitControls(camera, renderer.domElement)
  : null;
world.add(new THREE.HemisphereLight(0xffffff, 0x708369, 2.7));
const light = new THREE.DirectionalLight(0xffffff, 3);
light.position.set(-1, -2, 4);
world.add(light);
let group = new THREE.Group();
world.add(group);
function resetCamera() {
  const d = Math.max(scene?.table.width || 1.2, scene?.table.depth || 0.8);
  camera.position.set(d * 0.85, -d * 1.1, d * 0.95);
  if (controls) {
    controls.target.set(0, 0, 0.08);
    controls.update();
  }
}
resetCamera();
if (renderer) {
  new ResizeObserver(() => {
    const p = $("viewport");
    renderer.setSize(p.clientWidth, p.clientHeight);
    camera.aspect = p.clientWidth / p.clientHeight;
    camera.updateProjectionMatrix();
  }).observe($("viewport"));
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(world, camera);
  });
}
function dispose(node) {
  node.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) {
      for (const m of Array.isArray(o.material) ? o.material : [o.material])
        m.dispose();
    }
  });
}
function rebuild3D() {
  const generation = ++previewGeneration;
  world.remove(group);
  dispose(group);
  group = new THREE.Group();
  world.add(group);
  const table = scene?.table || { width: 1.2, depth: 0.8 };
  const tableMesh = new THREE.Mesh(
    new THREE.BoxGeometry(table.width, table.depth, 0.05),
    new THREE.MeshStandardMaterial({ color: "#becbbb", roughness: 0.9 }),
  );
  tableMesh.position.z = -0.025;
  group.add(tableMesh);
  const grid = new THREE.GridHelper(2, 20, 0x91aa94, 0xd0d9cc);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.051;
  group.add(grid);
  if (!scene) return;
  scene.objects.forEach((obj, index) => {
    if (!obj.selected || obj.geometry === "none") return;
    const material = new THREE.MeshStandardMaterial({
      color: colors[index % colors.length],
      roughness: 0.65,
      metalness: 0.05,
      transparent: obj.geometry === "proxy",
      opacity: obj.geometry === "proxy" ? 0.65 : 1,
    });
    const put = (node) => {
      if (generation !== previewGeneration) {
        dispose(node);
        return;
      }
      node.position.set(...obj.position);
      node.rotation.z = obj.yaw;
      node.userData.id = obj.id;
      group.add(node);
    };
    if (obj.mesh) {
      new OBJLoader().load(
        fileURL(obj.mesh),
        (node) => {
          node.traverse((child) => {
            if (child.isMesh) child.material = material;
          });
          node.scale.set(...obj.size);
          put(node);
        },
        undefined,
        () => notice(`无法加载 ${obj.name} 的网格`, "error"),
      );
    } else {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(...obj.size), material);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({
          color:
            obj.id === selected ? "#163d2b" : colors[index % colors.length],
        }),
      );
      mesh.add(edges);
      put(mesh);
    }
  });
}
rebuild3D();

const canvas = $("imageCanvas"),
  ctx = canvas.getContext("2d");
function draw() {
  if (!reference) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(reference, 0, 0);
  scene.objects.forEach((obj, index) => {
    const mask = masks.get(obj.id);
    if (mask) {
      ctx.globalAlpha = obj.selected
        ? obj.id === selected
          ? 0.48
          : 0.22
        : 0.06;
      ctx.drawImage(mask, 0, 0);
      ctx.globalAlpha = 1;
    }
    if (obj.id === selected || (!mask && obj.selected)) {
      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = Math.max(2, canvas.width / 500);
      ctx.strokeRect(
        obj.bbox[0],
        obj.bbox[1],
        obj.bbox[2] - obj.bbox[0],
        obj.bbox[3] - obj.bbox[1],
      );
    }
  });
  if (box) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 5]);
    ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    ctx.setLineDash([]);
  }
  for (const [x, y, label] of points) {
    ctx.beginPath();
    ctx.arc(x, y, Math.max(5, canvas.width / 180), 0, Math.PI * 2);
    ctx.fillStyle = label ? "#1bdc8e" : "#ff6c61";
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  const corners = tablePoints.length ? tablePoints : scene.table.corners;
  if (corners.length) {
    ctx.strokeStyle = "#f5d15c";
    ctx.lineWidth = 3;
    ctx.beginPath();
    corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (corners.length === 4) ctx.closePath();
    ctx.stroke();
    corners.forEach(([x, y], i) => {
      ctx.fillStyle = "#f5d15c";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 7);
      ctx.fill();
      ctx.font = `${Math.max(16, canvas.width / 55)}px sans-serif`;
      ctx.fillText(String(i + 1), x + 10, y - 9);
    });
  }
}
const imageLoad = (url) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
async function loadImages() {
  const sid = scene.id;
  reference = await imageLoad(fileURL(scene.image));
  if (scene.id !== sid) return;
  canvas.width = reference.width;
  canvas.height = reference.height;
  $("emptyImage").style.display = "none";
  masks.clear();
  draw();
  await Promise.all(
    scene.objects
      .filter((o) => o.mask)
      .map(async (obj, index) => {
        try {
          const image = await imageLoad(fileURL(obj.mask));
          if (scene.id !== sid) return;
          const c = document.createElement("canvas");
          c.width = canvas.width;
          c.height = canvas.height;
          const context = c.getContext("2d");
          context.drawImage(image, 0, 0);
          const pixels = context.getImageData(0, 0, c.width, c.height);
          const hex = colors[scene.objects.indexOf(obj) % colors.length];
          const color = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
          for (let p = 0; p < pixels.data.length; p += 4) {
            const alpha = pixels.data[p];
            pixels.data[p] = color[0];
            pixels.data[p + 1] = color[1];
            pixels.data[p + 2] = color[2];
            pixels.data[p + 3] = alpha;
          }
          context.putImageData(pixels, 0, 0);
          masks.set(obj.id, c);
          draw();
        } catch {
          notice(`掩码加载失败：${obj.name}`, "error");
        }
      }),
  );
}
function point(event) {
  const r = canvas.getBoundingClientRect();
  return [
    Math.max(
      0,
      Math.min(
        canvas.width - 1,
        ((event.clientX - r.left) * canvas.width) / r.width,
      ),
    ),
    Math.max(
      0,
      Math.min(
        canvas.height - 1,
        ((event.clientY - r.top) * canvas.height) / r.height,
      ),
    ),
  ];
}
canvas.addEventListener("pointerdown", (event) => {
  if (!scene || busy) return;
  const [x, y] = point(event);
  if (mode === "box") {
    start = [x, y];
    box = [x, y, x, y];
    canvas.setPointerCapture(event.pointerId);
  } else if (mode === "positive" || mode === "negative") {
    points.push([x, y, mode === "positive" ? 1 : 0]);
    draw();
  } else if (mode === "table") {
    tablePoints.push([x, y]);
    if (tablePoints.length === 4) {
      scene.table.corners = tablePoints;
      scene.table.calibration_source = "user";
      tablePoints = [];
      setMode("select");
      change();
      renderCalibration();
      notice("桌面四角已标注，请确认桌宽和桌深后保存。");
    }
    draw();
  } else {
    const candidates = scene.objects.filter(
      (o) =>
        x >= o.bbox[0] && x <= o.bbox[2] && y >= o.bbox[1] && y <= o.bbox[3],
    );
    candidates.sort(
      (a, b) =>
        (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]) -
        (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]),
    );
    if (candidates.length) select(candidates[0].id);
  }
});
canvas.addEventListener("pointermove", (event) => {
  if (start) {
    const [x, y] = point(event);
    box = [
      Math.min(x, start[0]),
      Math.min(y, start[1]),
      Math.max(x, start[0]),
      Math.max(y, start[1]),
    ];
    draw();
  }
});
canvas.addEventListener("pointerup", () => {
  start = null;
});
function setMode(next) {
  mode = next;
  document
    .querySelectorAll("[data-mode]")
    .forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  if (mode === "table") {
    tablePoints = [];
    notice("依次点击桌面的：远左 → 远右 → 近右 → 近左。");
  } else if (mode === "box")
    notice("拖动框选目标。当前选中物体会被修正；点击“新物体提示”可新增实例。");
}
document
  .querySelectorAll("[data-mode]")
  .forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$("clearPrompts").onclick = () => {
  box = null;
  points = [];
  tablePoints = [];
  draw();
};
$("newSelection").onclick = () => {
  selected = null;
  box = null;
  points = [];
  renderObjects();
  renderProperties();
  draw();
  setMode("box");
  notice("新物体模式：框选后点击“分割 / 修正选中物体”。");
};

function select(id) {
  selected = id;
  box = null;
  points = [];
  renderObjects();
  renderProperties();
  draw();
  rebuild3D();
}
function renderObjects() {
  $("objectCount").textContent =
    `${scene.objects.filter((o) => o.selected).length} / ${scene.objects.length} 已选`;
  $("objects").innerHTML = scene.objects.length
    ? scene.objects
        .map(
          (obj, i) =>
            `<div class="object ${obj.id === selected ? "active" : ""}" data-id="${obj.id}"><input type="checkbox" aria-label="保留 ${escape(obj.name)}" ${obj.selected ? "checked" : ""}><span class="swatch" style="background:${colors[i % colors.length]}"></span><div class="object-info"><div class="object-name">${escape(obj.name)}</div><div class="object-sub">${geometryLabels[obj.geometry]} · ${annotationLabels[obj.annotation]}</div></div><span class="object-id">${obj.score == null ? "" : (obj.score * 100).toFixed(0) + "%"}</span></div>`,
        )
        .join("")
    : '<div class="empty compact">暂无候选，请运行分割。</div>';
  document.querySelectorAll(".object").forEach((row) => {
    row.onclick = (e) => {
      if (e.target.type === "checkbox") {
        scene.objects.find((o) => o.id === row.dataset.id).selected =
          e.target.checked;
        change();
        draw();
        rebuild3D();
        $("objectCount").textContent =
          `${scene.objects.filter((o) => o.selected).length} / ${scene.objects.length} 已选`;
      } else select(row.dataset.id);
    };
  });
}
function renderProperties() {
  const obj = current();
  if (!obj) {
    $("properties").innerHTML =
      '<p class="muted">选择一个物体，设置尺寸、支撑物和类型标注。</p>';
    return;
  }
  const input = (label, key, value, extra = "") =>
    `<label>${label}<input data-key="${key}" value="${escape(value)}" ${extra}></label>`;
  $("properties").innerHTML =
    input("名称", "name", obj.name) +
    `<div class="two-col"><label>类型标注<select data-key="annotation">${Object.entries(
      annotationLabels,
    )
      .map(
        ([key, value]) =>
          `<option value="${key}" ${key === obj.annotation ? "selected" : ""}>${value}</option>`,
      )
      .join(
        "",
      )}</select></label><label>支撑物<select data-key="support"><option value="table">桌面</option>${scene.objects
      .filter((o) => o.id !== obj.id)
      .map(
        (o) =>
          `<option value="${o.id}" ${o.id === obj.support ? "selected" : ""}>${escape(o.name)}</option>`,
      )
      .join(
        "",
      )}</select></label></div><div class="three-col">${["宽 X / m", "深 Y / m", "高 Z / m"].map((name, i) => input(name, `size.${i}`, obj.size[i].toFixed(4), 'type="number" step="0.01" min="0.001" max="10"')).join("")}</div><div class="three-col">${["位置 X", "位置 Y", "位置 Z"].map((name, i) => input(name, `position.${i}`, obj.position[i].toFixed(4), 'type="number" step="0.01"')).join("")}</div><div class="three-col">${input("朝向 / 度", "yaw", (obj.yaw * 180) / Math.PI, 'type="number" step="5"')}${input("质量 / kg", "mass", obj.mass, 'type="number" step="0.01" min="0.001"')}${input("滑动摩擦", "friction", obj.friction, 'type="number" step="0.05" min="0" max="3"')}</div><label>参数来源<select data-key="parameter_source">${Object.entries(
      { assumed: "估计先验", user: "用户设定", measured: "实际测量" },
    )
      .map(
        ([key, value]) =>
          `<option value="${key}" ${key === obj.parameter_source ? "selected" : ""}>${value}</option>`,
      )
      .join(
        "",
      )}</select></label><label class="inline-check"><input type="checkbox" data-key="locked" ${obj.locked ? "checked" : ""}>锁定布局编辑（仿真中仍可运动）</label><label>备注<textarea data-key="notes" rows="2">${escape(obj.notes)}</textarea></label><label class="upload mesh-upload">导入此物体的网格<input id="meshFile" type="file" accept=".obj,.glb,.stl,.ply" hidden></label><p class="smallnote">导入后按当前宽度缩放，请核对尺寸。碰撞使用凸包；特殊结构只保存标注。</p>`;
  $("properties")
    .querySelectorAll("[data-key]")
    .forEach(
      (field) =>
        (field.onchange = () => {
          const key = field.dataset.key;
          let value =
            field.type === "checkbox"
              ? field.checked
              : field.type === "number"
                ? Number(field.value)
                : field.value;
          if (typeof value === "number" && !Number.isFinite(value)) return;
          if (key.includes(".")) {
            const [base, index] = key.split(".");
            obj[base][Number(index)] = value;
          } else obj[key] = key === "yaw" ? (value * Math.PI) / 180 : value;
          change();
          renderObjects();
          rebuild3D();
        }),
    );
  $("meshFile").onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      await save();
      const form = new FormData();
      form.append("file", file);
      scene = await api(`/api/projects/${scene.id}/objects/${obj.id}/mesh`, {
        method: "POST",
        body: form,
      });
      render();
      notice("网格已导入。请核对单位、尺寸与朝向。");
    } catch (error) {
      notice(error.message, "error");
    }
  };
}
function renderCalibration() {
  $("calibrationStatus").textContent = scene.table.corners.length
    ? "已有四角标注；尺寸由用户指定。框底部仍是接触位置近似。"
    : "未标定：使用整幅图片近似映射到桌面。";
}
function renderReport() {
  $("download").classList.toggle("disabled", !scene?.reports.export || dirty);
  if (scene?.reports.export)
    $("download").href = `/api/projects/${scene.id}/download`;
  const report = scene?.reports.validation;
  if (dirty) {
    $("report").innerHTML =
      '<p class="muted">参数已修改，旧检查结果不适用于当前编辑。请保存后重新检查。</p>';
    return;
  }
  if (!report) {
    $("report").innerHTML = '<p class="muted">尚无当前版本的检查结果。</p>';
    return;
  }
  $("report").innerHTML =
    `<strong class="${report.physics_check_pass ? "status-pass" : "status-fail"}">${report.physics_check_pass ? "短时物理检查通过" : "物理检查发现问题"}</strong><div class="metric"><span>仿真时长</span><span>${report.simulated_seconds.toFixed(2)} s</span></div><div class="metric"><span>初始最大穿插</span><span>${(report.initial_max_penetration_m * 1000).toFixed(2)} mm</span></div><div class="metric"><span>最大位移</span><span>${(Math.max(0, ...report.objects.map((o) => o.translation_m)) * 1000).toFixed(1)} mm</span></div><ul>${report.objects.map((o) => `<li class="${o.stable ? "status-pass" : "status-fail"}">${escape(o.name)} · ${(o.translation_m * 1000).toFixed(1)} mm · ${o.expected_support_contact ? "支撑接触存在" : "未保持预期支撑"}</li>`).join("")}</ul><p class="smallnote">仅验证当前刚体近似。视觉一致性未验证，不代表真实物理属性或任务成功。</p>`;
}
function render() {
  $("sceneTitle").textContent = scene.name;
  $("saveState").textContent = `已保存 · 修订 ${scene.revision}`;
  $("imageSize").textContent = `${scene.image_size.join(" × ")}`;
  $("tableWidth").value = scene.table.width;
  $("tableDepth").value = scene.table.depth;
  renderCalibration();
  renderObjects();
  renderProperties();
  renderReport();
  rebuild3D();
  draw();
}
for (const [id, key] of [
  ["tableWidth", "width"],
  ["tableDepth", "depth"],
])
  $(id).onchange = () => {
    if (!scene) return;
    scene.table[key] = Number($(id).value);
    scene.table.calibration_source = "user";
    change();
    rebuild3D();
  };
async function save() {
  if (!scene) throw new Error("请先上传图片");
  if (!dirty) return;
  scene = await api(`/api/projects/${scene.id}`, json("PUT", scene));
  dirty = false;
  render();
}
$("save").onclick = () =>
  save()
    .then(() => notice("修改已保存。"))
    .catch((error) => notice(error.message, "error"));
async function refreshProjects() {
  const list = await api("/api/projects");
  $("projects").innerHTML =
    '<option value="">选择项目</option>' +
    list
      .map(
        (p) =>
          `<option value="${p.id}">${escape(p.name)} · ${p.objects} 物体</option>`,
      )
      .join("");
  if (scene) $("projects").value = scene.id;
}
async function openProject(id) {
  const result = await api(`/api/projects/${id}`);
  scene = result.scene;
  dirty = false;
  selected = scene.objects[0]?.id || null;
  box = null;
  points = [];
  tablePoints = [];
  localStorage.setItem("r2s-project", id);
  render();
  await loadImages();
  resetCamera();
  if (result.job) poll(result.job);
}
$("projects").onchange = async () => {
  try {
    await saveIfPresent();
    if ($("projects").value) await openProject($("projects").value);
  } catch (error) {
    notice(error.message, "error");
  }
};
async function saveIfPresent() {
  if (scene && dirty) await save();
}
$("upload").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    await saveIfPresent();
    const data = new FormData();
    data.append("file", file);
    const created = await api("/api/projects", { method: "POST", body: data });
    await openProject(created.id);
    await refreshProjects();
    notice("图片已保存。开始自动分割，或框选你需要的物体。");
  } catch (error) {
    notice(error.message, "error");
  } finally {
    event.target.value = "";
  }
};
function setBusy(value) {
  busy = value;
  document
    .querySelectorAll(
      ".workspace button,.workspace input,.workspace select,.workspace textarea,#projects,#upload",
    )
    .forEach((el) => (el.disabled = value));
  $("reconstruct").disabled = value || !modelStatus.sam3d?.configured;
  $("segmentText").disabled = value || !modelStatus.sam3?.configured;
  $("segmentInstance").disabled = value || !modelStatus.sam3?.configured;
}
async function poll(job) {
  setBusy(true);
  notice(`正在执行 ${job.kind}，结果和日志会保存在当前项目。`, "busy");
  try {
    while (job.state === "queued" || job.state === "running") {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.state === "failed") throw new Error(job.error);
    const result = await api(`/api/projects/${job.scene_id}`);
    scene = result.scene;
    dirty = false;
    render();
    await loadImages();
    await refreshProjects();
    const detail =
      job.kind === "segment"
        ? `找到 ${scene.reports.segmentation?.candidate_count || 0} 个候选。请检查掩码并选择需要的实例。`
        : job.kind === "proxy"
          ? "占位预演已生成。这些盒子不是照片重建资产。"
          : job.kind === "validate"
            ? scene.reports.validation.physics_check_pass
              ? "短时物理检查通过；仍需视觉对照。"
              : "检查发现问题，请查看右侧逐物体结果。"
            : "操作完成，结果已保存。";
    notice(detail);
  } catch (error) {
    notice(error.message, "error");
  } finally {
    setBusy(false);
  }
}
async function submitAction(action) {
  try {
    await save();
    const job = await api(
      `/api/projects/${scene.id}/actions/${action}`,
      json("POST", { seconds: Number($("seconds").value) }),
    );
    await poll(job);
  } catch (error) {
    notice(error.message, "error");
  }
}
document
  .querySelectorAll("[data-action]")
  .forEach(
    (button) => (button.onclick = () => submitAction(button.dataset.action)),
  );
$("reconstruct").onclick = () => submitAction("reconstruct");
$("segmentText").onclick = async () => {
  try {
    await save();
    const labels = $("labels")
      .value.split(/[,，]/)
      .map((v) => v.trim())
      .filter(Boolean);
    await poll(
      await api(
        `/api/projects/${scene.id}/segment`,
        json("POST", { mode: "text", labels }),
      ),
    );
  } catch (error) {
    notice(error.message, "error");
  }
};
$("segmentInstance").onclick = async () => {
  try {
    await save();
    await poll(
      await api(
        `/api/projects/${scene.id}/segment`,
        json("POST", {
          mode: "instance",
          box: box || current()?.bbox || null,
          points,
          object_id: selected,
          label: "选中物体",
        }),
      ),
    );
    box = null;
    points = [];
    draw();
  } catch (error) {
    notice(error.message, "error");
  }
};
$("manualRegion").onclick = async () => {
  try {
    await save();
    if (!box) throw new Error("请先框选一个区域");
    scene = await api(
      `/api/projects/${scene.id}/region`,
      json("POST", { box, label: "人工框选物体" }),
    );
    selected = scene.objects.at(-1).id;
    box = null;
    points = [];
    dirty = false;
    render();
    notice("已保存人工区域，没有运行分割模型。可以导入网格或生成占位预演。");
  } catch (error) {
    notice(error.message, "error");
  }
};
$("resetView").onclick = resetCamera;
$("help").onclick = () => $("helpDialog").showModal();
$("closeHelp").onclick = () => $("helpDialog").close();
window.addEventListener("beforeunload", (event) => {
  if (dirty) {
    event.preventDefault();
    event.returnValue = "";
  }
});
async function boot() {
  try {
    modelStatus = await api("/api/status");
    $("modelStatus").innerHTML = ["sam3", "sam3d"]
      .map(
        (key) =>
          `<div class="model"><span class="dot ${modelStatus[key].configured ? "ok" : ""}"></span><div>${key === "sam3" ? "SAM3" : "SAM 3D Objects"}<small>${modelStatus[key].configured ? "环境已配置" : key === "sam3d" ? "等待权重权限与运行环境" : "未配置本地模型"}</small></div></div>`,
      )
      .join("");
    setBusy(false);
    await refreshProjects();
    const id = localStorage.getItem("r2s-project");
    if (id) {
      try {
        await openProject(id);
        $("projects").value = id;
      } catch {
        localStorage.removeItem("r2s-project");
      }
    }
  } catch (error) {
    notice(error.message, "error");
  }
}
boot();
