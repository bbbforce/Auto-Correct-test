"""
ParaView OsMesa 离屏渲染脚本。
由 simulation_executor_agent 通过 subprocess 调用 pvpython 执行。

用法:
    pvpython paraview_render.py <result_dir>

输出:
    将渲染结果保存为 pv_<原文件名>.png，并将路径以 JSON 列表输出到 stdout。
"""

import os
import sys
import json
import glob

from paraview.simple import *

# ─── 配置 ───────────────────────────────────────────────
RESOLUTION = [1920, 1080]
BG_COLOR = [0.12, 0.12, 0.18]

# 根据 XDMF 文件名关键词推断物理量名称和色图
FIELD_HINTS = {
    "displacement": {"title": "Displacement Magnitude", "preset": "Cool to Warm"},
    "stress":       {"title": "Von Mises Stress (Pa)",  "preset": "Cool to Warm"},
    "von_mises":    {"title": "Von Mises Stress (Pa)",  "preset": "Cool to Warm"},
    "temperature":  {"title": "Temperature (K)",        "preset": "Black-Body Radiation"},
    "velocity":     {"title": "Velocity Magnitude",     "preset": "Rainbow Uniform"},
    "pressure":     {"title": "Pressure (Pa)",          "preset": "Cool to Warm"},
    "damage":       {"title": "Damage Field d",         "preset": "Cool to Warm"},
    "phase":        {"title": "Phase Field",            "preset": "Cool to Warm"},
}


def infer_field_info(xdmf_basename):
    """从 XDMF 文件名推断物理量的标题和色图。"""
    name_lower = xdmf_basename.lower()
    for keyword, info in FIELD_HINTS.items():
        if keyword in name_lower:
            return info
    return {"title": xdmf_basename.replace(".xdmf", ""), "preset": "Cool to Warm"}


def detect_geometry_dim(reader):
    """检测几何维度（2D 还是 3D），用于设置相机角度。"""
    bounds = reader.GetDataInformation().GetBounds()
    # bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    z_range = bounds[5] - bounds[4]
    # 如果 Z 范围为 0 或极小，认为是 2D 问题
    if abs(z_range) < 1e-10:
        return 2
    return 3


def render_xdmf(xdmf_path, output_dir):
    """渲染单个 XDMF 文件，返回输出 PNG 路径。若失败返回 None。"""
    basename = os.path.basename(xdmf_path)
    name_stem = basename.replace(".xdmf", "")
    output_png = os.path.join(output_dir, f"pv_{name_stem}.png")

    try:
        # 读取 XDMF
        reader = XDMFReader(FileNames=[xdmf_path])
        reader.UpdatePipeline()

        info = reader.GetDataInformation()
        n_points = info.GetNumberOfPoints()
        n_cells = info.GetNumberOfCells()

        if n_points == 0:
            print(f"[SKIP] {basename}: empty dataset", file=sys.stderr)
            return None

        # 检测数据维度
        geo_dim = detect_geometry_dim(reader)

        # 获取物理量信息
        pd = info.GetPointDataInformation()
        cd = info.GetCellDataInformation()

        # 推断字段信息
        field_info = infer_field_info(basename)

        # 创建渲染视图
        rv = CreateRenderView()
        rv.ViewSize = RESOLUTION
        rv.Background = BG_COLOR

        # 显示数据
        display = Show(reader, rv)
        display.Representation = "Surface"

        # 着色：优先使用点数据，其次使用单元数据
        colored = False
        if pd.GetNumberOfArrays() > 0:
            arr = pd.GetArrayInformation(0)
            arr_name = arr.GetName()
            n_comp = arr.GetNumberOfComponents()

            if n_comp > 1:
                # 矢量场：按 Magnitude 着色
                ColorBy(display, ("POINTS", arr_name, "Magnitude"))
            else:
                ColorBy(display, ("POINTS", arr_name))

            ctf = GetColorTransferFunction(arr_name)
            ctf.ApplyPreset(field_info["preset"], True)
            display.RescaleTransferFunctionToDataRange(True, False)

            # 标量条
            sb = GetScalarBar(ctf, rv)
            sb.Title = field_info["title"]
            sb.ComponentTitle = ""
            sb.Visibility = 1
            sb.TitleFontSize = 18
            sb.LabelFontSize = 14
            colored = True

        elif cd.GetNumberOfArrays() > 0:
            arr = cd.GetArrayInformation(0)
            arr_name = arr.GetName()
            ColorBy(display, ("CELLS", arr_name))
            ctf = GetColorTransferFunction(arr_name)
            ctf.ApplyPreset(field_info["preset"], True)
            display.RescaleTransferFunctionToDataRange(True, False)
            sb = GetScalarBar(ctf, rv)
            sb.Title = field_info["title"]
            sb.Visibility = 1
            colored = True

        # 对 3D 问题：添加边线让网格结构可见
        if geo_dim == 3:
            display.Representation = "Surface With Edges"
            display.EdgeColor = [0.3, 0.3, 0.3]
            display.LineWidth = 0.5

        # 相机设置
        rv.ResetCamera()

        if geo_dim == 3:
            # 3D：稍微旋转一个角度，获得等轴测视角
            camera = rv.GetActiveCamera()
            camera.Azimuth(30)
            camera.Elevation(20)
            rv.ResetCamera()

        # 渲染并保存
        Render()
        SaveScreenshot(output_png, rv, ImageResolution=RESOLUTION)

        # 清理当前 pipeline
        Delete(display)
        Delete(reader)
        Delete(rv)

        print(f"[OK] {basename} -> {output_png} "
              f"(points={n_points}, cells={n_cells}, dim={geo_dim}D)",
              file=sys.stderr)
        return output_png

    except Exception as e:
        print(f"[ERROR] {basename}: {e}", file=sys.stderr)
        # 尝试清理
        try:
            Delete()
        except Exception:
            pass
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: pvpython paraview_render.py <result_dir>", file=sys.stderr)
        sys.exit(1)

    result_dir = sys.argv[1]

    if not os.path.isdir(result_dir):
        print(f"Directory not found: {result_dir}", file=sys.stderr)
        sys.exit(1)

    # 扫描所有 XDMF 文件
    xdmf_files = sorted(glob.glob(os.path.join(result_dir, "*.xdmf")))

    if not xdmf_files:
        print("No XDMF files found.", file=sys.stderr)
        # 输出空列表
        print(json.dumps([]))
        sys.exit(0)

    # 过滤：只渲染有配对 .h5 文件的 XDMF
    valid_files = []
    for xf in xdmf_files:
        # 检查 XDMF 是否引用了 H5 文件
        try:
            with open(xf, "r") as f:
                content = f.read()
            # 找到引用的 h5 文件名
            import re
            h5_refs = re.findall(r'>([^<>"]+\.h5):', content)
            if h5_refs:
                xdmf_dir = os.path.dirname(xf)
                all_h5_exist = all(
                    os.path.exists(os.path.join(xdmf_dir, h5_ref))
                    for h5_ref in h5_refs
                )
                if all_h5_exist:
                    valid_files.append(xf)
                else:
                    missing = [h for h in h5_refs
                               if not os.path.exists(os.path.join(xdmf_dir, h))]
                    print(f"[SKIP] {os.path.basename(xf)}: missing H5 files: {missing}",
                          file=sys.stderr)
            else:
                # 内联数据，可以直接渲染
                valid_files.append(xf)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(xf)}: cannot parse: {e}", file=sys.stderr)

    if not valid_files:
        print("No renderable XDMF files (H5 data missing).", file=sys.stderr)
        print(json.dumps([]))
        sys.exit(0)

    print(f"Found {len(valid_files)} renderable XDMF file(s).", file=sys.stderr)

    # 逐个渲染
    rendered_files = []
    for xf in valid_files:
        result = render_xdmf(xf, result_dir)
        if result:
            rendered_files.append(result)

    # 输出结果到 stdout（JSON 格式）
    print(json.dumps(rendered_files))


if __name__ == "__main__":
    main()
