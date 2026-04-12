import os
import json
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

from pipeline import run_pipeline
from error_memory import ErrorMemory
app = FastAPI(title="agent协同仿真工作台")

# 静态文件
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/files/runs", StaticFiles(directory=RUNS_DIR), name="runs_files")
app.mount("/files/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads_files")

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页面。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """接收上传文件，保存到临时目录，返回文件路径列表。"""
    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(UPLOADS_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    saved_paths = []
    for f in files:
        file_path = os.path.join(session_dir, f.filename)
        content = await f.read()
        with open(file_path, "wb") as fp:
            fp.write(content)
        saved_paths.append(file_path)

    return {"files": saved_paths, "session_id": session_id}

@app.get("/api/history")
async def get_history():
    """获取所有历史仿真任务。按时间倒序排列。"""
    runs = []
    if os.path.exists(RUNS_DIR):
        for run_id in os.listdir(RUNS_DIR):
            if run_id.startswith("run_"):
                full_path = os.path.join(RUNS_DIR, run_id)
                if os.path.isdir(full_path):
                    parts = run_id.replace('run_', '').split('_')
                    if len(parts) == 2:
                        readable = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]} {parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:]}"
                    else:
                        readable = run_id
                    runs.append({
                        "id": run_id,
                        "title": f"仿真分析 {readable}"
                    })
    runs.sort(key=lambda x: x['id'], reverse=True)
    return JSONResponse(content={"history": runs})

@app.get("/api/history/{run_id}")
async def get_history_detail(run_id: str):
    """获取某一次历史仿真的最终报告及相关源文件、图片。"""
    run_path = os.path.join(RUNS_DIR, run_id)
    result_path = os.path.join(run_path, "result")
    report_file = os.path.join(run_path, "simulation_report.txt")
    
    content = "*当前任务历史不存在最终报告，说明任务可能由于异常或被强行中断而失败。*"
    if os.path.isfile(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
            
    images = []
    files = []
    
    def scan_files(dir_path, is_result_dir=False):
        if not os.path.isdir(dir_path):
            return
        for fname in os.listdir(dir_path):
            if fname.startswith("."): 
                continue
            if not is_result_dir and fname in ["simulation_report.txt"]:
                continue
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath):
                base_url = f"/files/runs/{run_id}/result/{fname}" if is_result_dir else f"/files/runs/{run_id}/{fname}"
                ext = fname.split('.')[-1].lower() if '.' in fname else ''
                if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                    images.append({"name": fname, "url": base_url})
                else:
                    files.append({"name": fname, "url": base_url})

    scan_files(run_path, is_result_dir=False)
    scan_files(result_path, is_result_dir=True)
    
    events_file = os.path.join(run_path, "ui_events.json")
    ui_events = []
    if os.path.isfile(events_file):
        try:
            with open(events_file, "r", encoding="utf-8") as f:
                ui_events = json.load(f)
        except Exception:
            pass
    
    return JSONResponse(content={
        "report": content,
        "images": images,
        "files": files,
        "events": ui_events
    })

@app.delete("/api/history/{run_id}")
async def delete_history(run_id: str):
    """人工删除某一次仿真历史及物理文件。"""
    if not run_id.startswith("run_"):
        return JSONResponse(status_code=400, content={"message": "非法目录访问"})
    
    run_path = os.path.join(RUNS_DIR, run_id)
    if os.path.exists(run_path) and os.path.isdir(run_path):
        try:
            shutil.rmtree(run_path)
            return JSONResponse(content={"success": True, "message": f"{run_id} 删除成功"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": str(e)})
    return JSONResponse(status_code=404, content={"message": "历史记录未找到"})

@app.get("/api/error_memory")
async def get_error_memory():
    """获取错误知识库的内容及统计。"""
    try:
        em = ErrorMemory()
        entries = em.entries
        stats = em.get_stats()
        # 按照出现次数倒序
        sorted_entries = sorted(entries, key=lambda x: -x.get("occurrences", 1))
        return JSONResponse(content={"entries": sorted_entries, "stats": stats})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.delete("/api/error_memory/{entry_id}")
async def delete_error_memory_entry(entry_id: str):
    """删除指定的错误知识条目。"""
    try:
        em = ErrorMemory()
        success = em.prune(entry_id)
        if success:
            return JSONResponse(content={"success": True, "message": "删除成功"})
        else:
            return JSONResponse(status_code=404, content={"message": "尚未找到该条目"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.websocket("/ws/simulate")
async def ws_simulate(websocket: WebSocket):
    """WebSocket 仿真端点：接收参数，支持响应中断信号。"""
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        files = data.get("files", [])
        max_retries = data.get("max_retries", 3)
        
        # 将用户开局的上传记录进录像头部
        run_events = [
            {"type": "user_input", "prompt": prompt, "files": files}
        ]

        async def ws_event_handler(event):
            run_events.append(event)
            try:
                await websocket.send_json(event)
            except Exception:
                pass

        loop = asyncio.get_event_loop()
        
        # 将 pipeline 挂进 asyncio 的任务池并发执行
        pipeline_task = loop.create_task(run_pipeline(
            prompt=prompt,
            files=files,
            max_retries=max_retries,
            on_event=ws_event_handler,
        ))

        # 中断信号监听任务
        async def listen_for_cancel():
            try:
                while True:
                    msg = await websocket.receive_text()
                    if msg == "CANCEL":
                        return "CANCEL"
            except WebSocketDisconnect:
                return "DISCONNECT"

        cancel_task = loop.create_task(listen_for_cancel())

        # 任何一个完成，就触发返回
        done, pending = await asyncio.wait(
            [pipeline_task, cancel_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        if cancel_task in done:
            # 用户中断或断联
            if not pipeline_task.done():
                pipeline_task.cancel()  # 真正地对内部发出中止异常 CancelledError
                err_ev = {"type": "pipeline_error", "message": "此任务已被人工强行终止。"}
                run_events.append(err_ev)
                try:
                    await websocket.send_json(err_ev)
                except Exception:
                    pass
        else:
            # 如果是流水线正常执行完毕或自发异常报错退出，则结束并清理挂起的侦听任务
            cancel_task.cancel()

        # 最后整理并保存日志（无论成功还是强制中断）
        run_dir = None
        for ev in run_events:
            # 兼容从各种消息中找 run_dir (可以是 info, 或 pipeline_complete)
            if ev.get("type") == "info" and str(ev.get("message", "")).startswith("运行目录: "):
                run_dir = ev["message"].replace("运行目录: ", "").strip()
                break
            elif ev.get("type") == "pipeline_complete" and "run_dir" in ev:
                run_dir = ev["run_dir"]
                break
                
        if run_dir and os.path.exists(run_dir):
            try:
                with open(os.path.join(run_dir, "ui_events.json"), "w", encoding="utf-8") as f:
                    json.dump(run_events, f, ensure_ascii=False)
            except Exception:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "pipeline_error", "message": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    print("\n agent协同仿真工作台")
    print("   http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
