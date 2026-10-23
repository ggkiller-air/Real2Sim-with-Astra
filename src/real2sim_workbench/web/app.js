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
  previewGeneration = 0,
  removed = [],
  thumbnails = new Map(),
  building = false;
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
    if (!p.clientWidth || !p.clientHeight) return;
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
    if (!obj.selected) return;
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
  const corners = [];
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
        o.selected &&
        x >= o.bbox[0] &&
        x <= o.bbox[2] &&
        y >= o.bbox[1] &&
        y <= o.bbox[3],
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
canvas.addEventListener("pointerup", async () => {
  const adding = !!start;
  start = null;
  if (adding && box && box[2] - box[0] > 4 && box[3] - box[1] > 4)
    await addRegion();
});
function setMode(next) {
  mode = next;
  canvas.classList.toggle("adding", mode === "box");
  $("cancelAdd").hidden = mode !== "box";
}
$("cancelAdd").onclick = () => {
  box = null;
  setMode("select");
  draw();
  notice("已取消添加。");
};
$("newSelection").onclick = () => {
  if (!scene || busy) return;
  selected = null;
  box = null;
  points = [];
  showView(false);
  setMode("box");
  draw();
  notice("在图片上框住遗漏的物体，松开后自动添加。");
};
async function addRegion() {
  const chosenBox = [...box];
  setMode("select");
  setBusy(true);
  try {
    await save();
    if (modelStatus.sam3?.configured) {
      await poll(
        await api(
          `/api/projects/${scene.id}/segment`,
          json("POST", {
            mode: "instance",
            box: chosenBox,
            label: "添加的物体",
          }),
        ),
      );
    } else {
      scene = await api(
        `/api/projects/${scene.id}/region`,
        json("POST", { box: chosenBox, label: "框选物体" }),
      );
      dirty = false;
      render();
      notice("已保存框选区域；本机 SAM3 尚未就绪。");
    }
    if (scene.reports.segmentation?.added_count === 0)
      notice("没有新增物体：目标可能已存在，或属于排除的机械臂区域。");
    selected = scene.objects.filter((o) => o.selected).at(-1)?.id;
    box = null;
    draw();
    renderObjects();
  } catch (error) {
    notice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function select(id) {
  selected = id;
  box = null;
  points = [];
  renderObjects();
  renderProperties();
  draw();
  rebuild3D();
}
const displayName = (name) =>
  ({
    "small object on the table": "桌面物体",
    "button box": "按钮盒",
    box: "盒子",
    tool: "工具",
    "paper cup": "纸杯",
    cup: "杯子",
    bowl: "碗",
    tray: "托盘",
    spoon: "勺子",
    bottle: "瓶子",
    jar: "罐子",
    glass: "玻璃杯",
    bag: "袋子",
    hammer: "锤子",
    brush: "刷子",
    fruit: "水果",
    button: "按钮",
    block: "方块",
    object: "物体",
    "push button": "按钮",
    "tea bag": "茶包",
    tongs: "夹子",
    "utensil rack": "器具架",
  })[name] || name;
function thumbnail(obj) {
  const key = `${scene.id}/${obj.id}/${obj.bbox.join()}`;
  if (thumbnails.has(key)) return thumbnails.get(key);
  if (!reference) return "";
  const c = document.createElement("canvas");
  c.width = 100;
  c.height = 80;
  const [x, y, x2, y2] = obj.bbox;
  const scale = Math.min(96 / (x2 - x), 76 / (y2 - y));
  c.getContext("2d").drawImage(
    reference,
    x,
    y,
    x2 - x,
    y2 - y,
    (100 - (x2 - x) * scale) / 2,
    (80 - (y2 - y) * scale) / 2,
    (x2 - x) * scale,
    (y2 - y) * scale,
  );
  const url = c.toDataURL();
  thumbnails.set(key, url);
  return url;
}
async function persist() {
  setBusy(true);
  try {
    await save();
  } catch (error) {
    notice(error.message, "error");
  } finally {
    setBusy(false);
  }
}
function renderObjects() {
  const objects = scene.objects.filter((o) => o.selected);
  $("objectCount").textContent = `${objects.length} 个`;
  $("undoRemove").hidden = !removed.length;
  $("objects").innerHTML = objects.length
    ? objects
        .map((obj) => {
          const weights = [...new Set([0.02, 0.1, 0.5, 1, obj.mass])].sort(
            (a, b) => a - b,
          );
          return `<article class="object-card ${obj.id === selected ? "active" : ""}" data-id="${obj.id}">
      <img class="thumb" src="${thumbnail(obj)}" alt="${escape(displayName(obj.name))}">
      <div class="object-main"><div class="object-title"><button class="object-name" title="查看物体详情">${escape(displayName(obj.name))}</button><button class="remove-object" aria-label="删除 ${escape(displayName(obj.name))}">×</button></div>
      <div class="card-options"><select class="object-type" aria-label="${escape(displayName(obj.name))} 类型">${Object.entries(
        annotationLabels,
      )
        .map(
          ([key, value]) =>
            `<option value="${key}" ${obj.annotation === key ? "selected" : ""}>${value.replace("（仅标注）", " · 标注")}</option>`,
        )
        .join(
          "",
        )}</select><select class="object-mass" aria-label="${escape(displayName(obj.name))} 重量">${weights.map((w) => `<option value="${w}" ${obj.mass === w ? "selected" : ""}>${w * 1000} 克${obj.parameter_source === "assumed" && obj.mass === w ? "（估）" : ""}</option>`).join("")}</select></div></div></article>`;
        })
        .join("")
    : '<div class="empty-list">没有保留的物体。可以自动补全，或框选添加。</div>';
  $("objects")
    .querySelectorAll(".object-card")
    .forEach((card) => {
      const obj = scene.objects.find((o) => o.id === card.dataset.id);
      card.querySelector(".thumb").onclick = () => {
        if (!busy) {
          select(obj.id);
          showView(false);
        }
      };
      card.querySelector(".object-name").onclick = () => {
        if (!busy) {
          select(obj.id);
          $("propertyDialog").showModal();
        }
      };
      card.querySelector(".remove-object").onclick = async () => {
        if (busy) return;
        const affected = scene.objects
          .filter((o) => o.support === obj.id)
          .map((o) => [o.id, o.support]);
        affected.forEach(
          ([id]) => (scene.objects.find((o) => o.id === id).support = "table"),
        );
        obj.selected = false;
        removed.push({ id: obj.id, affected });
        if (selected === obj.id) selected = null;
        change();
        draw();
        renderObjects();
        rebuild3D();
        await persist();
      };
      card.querySelector(".object-type").onchange = async (e) => {
        obj.annotation = e.target.value;
        change();
        await persist();
      };
      card.querySelector(".object-mass").onchange = async (e) => {
        obj.mass = Number(e.target.value);
        obj.parameter_source = "user";
        change();
        await persist();
      };
    });
}
$("undoRemove").onclick = async () => {
  const entry = removed.pop();
  if (!entry || busy) return;
  const obj = scene.objects.find((o) => o.id === entry.id);
  obj.selected = true;
  entry.affected.forEach(([id, support]) => {
    const child = scene.objects.find((o) => o.id === id);
    if (child) child.support = support;
  });
  if (
    obj.support !== "table" &&
    !scene.objects.find((o) => o.id === obj.support)?.selected
  )
    obj.support = "table";
  change();
  renderObjects();
  draw();
  rebuild3D();
  await persist();
};
function renderProperties() {
  const obj = current();
  if (!obj) {
    $("properties").innerHTML = "选择一个物体查看详情。";
    return;
  }
  const input = (label, key, value, extra = "") =>
    `<label>${label}<input data-key="${key}" value="${escape(value)}" ${extra}></label>`;
  $("properties").innerHTML =
    input("名称", "name", obj.name) +
    `<div class="three-col">${["宽 / m", "深 / m", "高 / m"].map((label, i) => input(label, `size.${i}`, obj.size[i], 'type="number" min="0.001" max="10" step="0.01"')).join("")}</div>` +
    `<div class="two-col">${input("质量 / kg", "mass", obj.mass, 'type="number" min="0.001" step="0.01"')}${input("摩擦系数", "friction", obj.friction, 'type="number" min="0" max="3" step="0.05"')}</div>` +
    `<label class="button upload">导入已有模型<input id="meshFile" type="file" accept=".obj,.glb,.stl,.ply" hidden></label><p class="smallnote">${geometryLabels[obj.geometry]} · 默认尺寸与重量为估计值。特殊类型仅标注，当前按普通刚体处理。</p>`;
  $("properties")
    .querySelectorAll("[data-key]")
    .forEach(
      (field) =>
        (field.onchange = async () => {
          const key = field.dataset.key,
            value = field.type === "number" ? Number(field.value) : field.value;
          if (key.includes(".")) {
            const [base, i] = key.split(".");
            obj[base][+i] = value;
          } else obj[key] = value;
          if (key !== "name") obj.parameter_source = "user";
          change();
          await persist();
        }),
    );
  $("meshFile").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setBusy(true);
    try {
      await save();
      const form = new FormData();
      form.append("file", file);
      scene = await api(`/api/projects/${scene.id}/objects/${obj.id}/mesh`, {
        method: "POST",
        body: form,
      });
      render();
      notice("模型已导入。");
    } catch (error) {
      notice(error.message, "error");
    } finally {
      setBusy(false);
    }
  };
}
function renderReport() {
  $("download").classList.toggle("disabled", !scene?.reports.export || dirty);
  if (scene?.reports.export)
    $("download").href = `/api/projects/${scene.id}/download`;
  $("download").hidden = !scene?.reports.export || dirty;
  $("showReport").hidden = !scene?.reports.validation || dirty;
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

  renderObjects();
  renderProperties();
  renderReport();
  rebuild3D();
  draw();
  setBusy(busy);
}
async function save() {
  if (!scene) throw new Error("请先上传图片");
  if (!dirty) return;
  scene = await api(`/api/projects/${scene.id}`, json("PUT", scene));
  dirty = false;
  render();
}
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
  reference = null;
  thumbnails.clear();
  removed = [];
  scene = result.scene;
  dirty = false;
  selected = scene.objects[0]?.id || null;
  box = null;
  points = [];
  tablePoints = [];
  localStorage.setItem("r2s-project", id);
  render();
  await loadImages();
  renderObjects();
  resetCamera();
  notice("点击物体可查看详情；删除多余物体，框选补充遗漏。");
  if (result.job) await poll(result.job);
  setBusy(false);
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
    await autoDetect();
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
      ".workspace button,.workspace select,.workspace input,#properties input,#properties select,#projects,#upload",
    )
    .forEach((el) => (el.disabled = value));
  $("autoDetect").disabled = value || !scene || !modelStatus.sam3?.configured;
  $("newSelection").disabled = value || !scene;
  $("buildScene").disabled = value || !scene?.objects.some((o) => o.selected);
}
async function poll(job) {
  setBusy(true);
  const jobNames = {
    segment: "正在识别物体",
    proxy: "正在准备占位几何",
    initialize: "正在安排物体位置",
    optimize: "正在减少物体穿插",
    validate: "正在检查物理稳定性",
    reconstruct: "正在生成三维资产",
  };
  notice(jobNames[job.kind] || "正在处理", "busy");
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
    renderObjects();
    await refreshProjects();
    const detail =
      job.kind === "segment"
        ? `新增 ${scene.reports.segmentation?.added_count || 0} 个物体；检测到的机械臂已排除。可删除多余项或框选补充。`
        : job.kind === "proxy"
          ? "占位预演已生成。这些盒子不是照片重建资产。"
          : job.kind === "validate"
            ? scene.reports.validation.physics_check_pass
              ? "短时物理检查通过；仍需视觉对照。"
              : "检查发现问题，点击“检查结果”查看详情。"
            : "操作完成，结果已保存。";
    notice(detail);
  } catch (error) {
    notice(error.message, "error");
    throw error;
  } finally {
    setBusy(building);
  }
}
async function autoDetect() {
  if (!scene) return;
  if (!modelStatus.sam3?.configured) {
    notice("本机 SAM3 尚未就绪，可以先框选添加。", "error");
    return;
  }
  setBusy(true);
  try {
    await save();
    await poll(
      await api(
        `/api/projects/${scene.id}/segment`,
        json("POST", { mode: "auto" }),
      ),
    );
  } catch (error) {
    notice(error.message, "error");
  } finally {
    setBusy(false);
  }
}
$("autoDetect").onclick = autoDetect;
$("buildScene").onclick = async () => {
  building = true;
  setBusy(true);
  try {
    await save();
    const hasMissing = scene.objects.some(
      (o) => o.selected && o.geometry === "none",
    );
    const actions = hasMissing
      ? [
          modelStatus.sam3d?.configured ? "reconstruct" : "proxy",
          "initialize",
          "optimize",
          "validate",
        ]
      : ["initialize", "optimize", "validate"];
    for (const action of actions)
      await poll(
        await api(
          `/api/projects/${scene.id}/actions/${action}`,
          json("POST", { seconds: 3 }),
        ),
      );
    showView(true);
    notice(
      scene.reports.validation.physics_check_pass
        ? "场景已生成，短时物理检查通过。可以下载 MuJoCo 场景。"
        : "场景已生成，但物理检查发现问题，请查看检查结果。",
      scene.reports.validation.physics_check_pass ? "" : "error",
    );
  } catch (error) {
    notice(error.message, "error");
  } finally {
    building = false;
    setBusy(false);
  }
};
function showView(preview) {
  $("imageWrap").hidden = preview;
  $("viewport").hidden = !preview;
  $("resetView").hidden = !preview;
  $("photoTab").classList.toggle("active", !preview);
  $("previewTab").classList.toggle("active", preview);
}
$("photoTab").onclick = () => showView(false);
$("previewTab").onclick = () => showView(true);
$("resetView").onclick = resetCamera;
$("help").onclick = () => $("helpDialog").showModal();
$("closeHelp").onclick = () => $("helpDialog").close();
$("closeProperties").onclick = () => $("propertyDialog").close();
$("showReport").onclick = () => $("reportDialog").showModal();
$("closeReport").onclick = () => $("reportDialog").close();
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
    $("buildScene").textContent = modelStatus.sam3d?.configured
      ? "生成场景"
      : "生成占位场景";
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
    setBusy(false);
  } catch (error) {
    notice(error.message, "error");
  }
}
boot();
