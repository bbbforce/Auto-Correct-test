/**
 * MCP-SIM 仿真工作台 — 前端逻辑 (ChatGPT 风格)
 */

// ── DOM 元素 ──────────────────────────────────────────
const chatOutput = document.getElementById('chat-output');
const welcomeMsg = document.getElementById('welcome-msg');
const promptInput = document.getElementById('prompt-input');
const actionBtn = document.getElementById('action-btn');
const attachBtn = document.getElementById('attach-btn');
const retriesSelect = document.getElementById('retries-select');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const fileChips = document.getElementById('file-chips');

const newChatBtn = document.getElementById('new-chat-btn');
const historyList = document.getElementById('history-list');
const mobileSidebarToggle = document.getElementById('mobile-sidebar-toggle');
const sidebar = document.querySelector('.sidebar');

const iconSend = actionBtn.querySelector('.icon-send');
const iconStop = actionBtn.querySelector('.icon-stop');

// ── 状态 ──────────────────────────────────────────────
let uploadedFiles = [];   // [{name, serverPath}]
let isRunning = false;
let ws = null;

// 在单次对话中跟踪当前的 Assistant DOM
let currentAssistantMsgInfo = {
    row: null,
    thinkingPanel: null,
    thinkContent: null,
    thinkIcon: null,
    responseBody: null,
    rawReportText: ""
};

// ── 初始化 ────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});

// ── 历史记录列 ────────────────────────────────────────
async function loadHistory() {
    try {
        const resp = await fetch('/api/history');
        const data = await resp.json();
        renderHistoryList(data.history);
    } catch (err) {
        console.error('加载历史失败:', err);
    }
}

function renderHistoryList(historyArray) {
    historyList.innerHTML = '';
    if (!historyArray || historyArray.length === 0) {
        historyList.innerHTML = '<div style="color:#A0A0A0; font-size:12px; padding:10px;">暂无历史记录</div>';
        return;
    }
    historyArray.forEach(item => {
        const itemBox = document.createElement('div');
        itemBox.className = 'history-item';
        
        const contentBox = document.createElement('div');
        contentBox.className = 'history-item-content';
        contentBox.textContent = item.title;
        contentBox.onclick = async () => {
            document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
            itemBox.classList.add('active');
            await loadHistoryDetail(item.id, item.title);
            if(window.innerWidth <= 768) {
                sidebar.classList.remove('open');
            }
        };

        const delBtn = document.createElement('button');
        delBtn.className = 'history-delete-btn';
        delBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M4 7H20M10 11V17M14 11V17M5 7L6 19C6 20.1046 6.89543 21 8 21H16C17.1046 21 18 20.1046 18 19L19 7M9 7V4C9 3.44772 9.44772 3 10 3H14C14.5523 3 15 3.44772 15 4V7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        delBtn.title = "删除";
        delBtn.onclick = async (e) => {
            e.stopPropagation();
            if (confirm(`确定要删除 ${item.title} 吗？\n删除后相关的物理文件与记录将无法恢复。`)) {
                try {
                    await fetch(`/api/history/${item.id}`, { method: 'DELETE' });
                    loadHistory(); // 重新拉取
                    // 如果删除的是当前处于 active 的任务，可能需要清空界面
                    if (itemBox.classList.contains('active')) {
                        chatOutput.innerHTML = '';
                        if (welcomeMsg) welcomeMsg.style.display = 'block';
                    }
                } catch(err) {
                    alert('删除失败:' + err);
                }
            }
        };
        
        itemBox.appendChild(contentBox);
        itemBox.appendChild(delBtn);
        historyList.appendChild(itemBox);
    });
}

async function loadHistoryDetail(runId, title) {
    // 保护：如果正在运行则不允许且提示
    if (isRunning) return;

    // 清理界面
    if (welcomeMsg) welcomeMsg.style.display = 'none';
    chatOutput.innerHTML = '';

    try {
        const resp = await fetch(`/api/history/${runId}`);
        const data = await resp.json();
        
        let hasUserInput = false;
        if (data.events && data.events.length > 0 && data.events[0].type === 'user_input') {
            hasUserInput = true;
            const ev = data.events[0];
            const originalFiles = (ev.files || []).map(p => {
                const nameParts = p.split(/[\\/]/);
                return { 
                    name: nameParts[nameParts.length - 1] || 'file',
                    serverPath: p 
                };
            });
            appendUserMessage(ev.prompt || '[发送了附件任务]', originalFiles);
        }

        if (!hasUserInput) {
            appendUserMessage(`查看历史记录：${title}`, []);
        }

        // 渲染骨架
        prepareAssistantMessage();
        
        // --- 对历史录像的新旧格式兼容 ---
        if (data.events && data.events.length > 0) {
            // "Squash" 流式合并压缩算法
            const squashedEvents = [];
            let currentStream = null;
            
            for (let ev of data.events) {
                if (ev.type === 'user_input') continue;
                
                if (ev.type === 'stream') {
                    if (currentStream && currentStream.step === ev.step) {
                        currentStream.chunk += ev.chunk;
                    } else {
                        currentStream = { type: 'stream', step: ev.step, chunk: ev.chunk };
                        squashedEvents.push(currentStream);
                    }
                } else {
                    currentStream = null;
                    squashedEvents.push(ev);
                }
            }
            
            // 一步到位顺序重放
            squashedEvents.forEach(ev => handleEvent(ev));
            
            // 回放完成将其默认折叠，免得占用大段空间
            if (currentAssistantMsgInfo.thinkingPanel) {
                currentAssistantMsgInfo.thinkingPanel.classList.remove('open');
            }
            if (currentAssistantMsgInfo.thinkIcon) {
                currentAssistantMsgInfo.thinkIcon.className = 'think-icon'; 
                currentAssistantMsgInfo.thinkIcon.textContent = '✔️';
            }
        } else {
            // 老旧版本的纯报告读取兜底
            currentAssistantMsgInfo.thinkingPanel.style.display = 'none'; // 隐藏 thinking
            
            // 1. 渲染文字报告
            currentAssistantMsgInfo.responseBody.innerHTML = marked.parse(data.report || '');
            currentAssistantMsgInfo.responseBody.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
            // 渲染历史报告中的公式
            if (window.MathJax) {
                MathJax.typesetPromise([currentAssistantMsgInfo.responseBody]).catch(err => console.error(err));
            }
        }
        
        // 2. 追加结果资产区块
        if ((data.images && data.images.length > 0) || (data.files && data.files.length > 0)) {
            const assetsDiv = document.createElement('div');
            assetsDiv.className = 'result-assets-container';
            
            // 图片流
            if (data.images && data.images.length > 0) {
                const imgSection = document.createElement('div');
                imgSection.innerHTML = `<div class="result-section-title">🖼️ 仿真结果图集</div>
                                        <div class="image-gallery"></div>`;
                const gallery = imgSection.querySelector('.image-gallery');
                data.images.forEach(img => {
                    gallery.innerHTML += `
                        <div class="image-card">
                            <a href="${img.url}" target="_blank">
                                <img src="${img.url}" alt="${img.name}" title="点击查看大图" loading="lazy">
                            </a>
                            <div class="image-card-title">${img.name}</div>
                        </div>
                    `;
                });
                assetsDiv.appendChild(imgSection);
            }
            
            // 附件流
            if (data.files && data.files.length > 0) {
                const fileSection = document.createElement('div');
                fileSection.style.marginTop = '16px';
                fileSection.innerHTML = `<div class="result-section-title">📁 附件源文件</div>
                                         <div class="files-gallery"></div>`;
                const gallery = fileSection.querySelector('.files-gallery');
                data.files.forEach(f => {
                    gallery.innerHTML += `
                        <a href="${f.url}" class="file-card" download="${f.name}">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 4v12m0 0l-4-4m4 4l4-4M4 20h16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
                            ${f.name}
                        </a>
                    `;
                });
                assetsDiv.appendChild(fileSection);
            }
            
            currentAssistantMsgInfo.responseBody.appendChild(assetsDiv);
        }
        
    } catch (err) {
        currentAssistantMsgInfo.thinkingPanel.style.display = 'none';
        currentAssistantMsgInfo.responseBody.innerHTML = `<span style="color:red">无法加载历史报告详情</span>`;
    }
    scrollToBottom();
}

newChatBtn.addEventListener('click', () => {
    if (isRunning) return;
    chatOutput.innerHTML = '';
    if (welcomeMsg) {
        chatOutput.appendChild(welcomeMsg);
        welcomeMsg.style.display = 'block';
    }
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    promptInput.value = '';
    uploadedFiles = [];
    renderFileChips();
    if(window.innerWidth <= 768) {
        sidebar.classList.remove('open');
    }
});

if (mobileSidebarToggle) {
    mobileSidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });
}

// ── 文件上传 ──────────────────────────────────────────
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

async function handleFiles(fileList) {
    if (!fileList.length) return;

    const formData = new FormData();
    for (const f of fileList) {
        formData.append('files', f);
    }

    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await resp.json();

        data.files.forEach((serverPath, i) => {
            const name = fileList[i].name;
            uploadedFiles.push({ name, serverPath });
            addFileChip(name, uploadedFiles.length - 1);
        });
    } catch (err) {
        console.error('上传失败:', err);
    }
    fileInput.value = '';
}

function addFileChip(name, index) {
    const chip = document.createElement('div');
    chip.className = 'mini-chip';
    const ext = name.split('.').pop().toLowerCase();
    const icon = { pdf:'📕', docx:'📘', xlsx:'📊', pptx:'📙', png:'🖼️', jpg:'🖼️', jpeg:'🖼️' }[ext] || '📄';
    chip.innerHTML = `<span>${icon} ${name}</span><span class="remove" data-idx="${index}">✕</span>`;
    chip.querySelector('.remove').addEventListener('click', () => {
        uploadedFiles.splice(index, 1);
        renderFileChips();
    });
    fileChips.appendChild(chip);
}

function renderFileChips() {
    fileChips.innerHTML = '';
    uploadedFiles.forEach((f, i) => addFileChip(f.name, i));
}

// ── 输入框自适应高度 ──────────────────────────────────
promptInput.addEventListener('input', () => {
    promptInput.style.height = 'auto';
    promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + 'px';
});

promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleActionClick();
    }
});

// ── 发送/停止 动作按钮 ────────────────────────────────
actionBtn.addEventListener('click', handleActionClick);

function handleActionClick() {
    if (isRunning) {
        // 如果正在运行，此时按钮代表停止
        stopSimulation();
    } else {
        // 如果空闲，此时按钮代表发送
        startSimulation();
    }
}

function stopSimulation() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        // 发送中断信号让后端处理 asyncio 取消
        ws.send("CANCEL");
        appendToThinking(`<span style="color:#D93025"><b>[人工中止]</b> 正在强行关闭仿真任务...</span>`);
    }
}

function startSimulation() {
    const prompt = promptInput.value.trim();
    const files = [...uploadedFiles];

    if (!prompt && files.length === 0) {
        promptInput.focus();
        return;
    }

    isRunning = true;
    
    // 切换按钮状态至 Stop
    actionBtn.classList.add('stop-active');
    iconSend.style.display = 'none';
    iconStop.style.display = 'block';

    // 清空工作区占位
    if (welcomeMsg) welcomeMsg.style.display = 'none';

    // 1. 渲染用户消息
    appendUserMessage(prompt, files);

    // 清空输入区
    promptInput.value = '';
    promptInput.style.height = 'auto';
    uploadedFiles = [];
    renderFileChips();

    // 2. 准备 Assistant 占位 DOM
    prepareAssistantMessage();

    // 建立 WebSocket 连接
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/simulate`);

    ws.onopen = () => {
        ws.send(JSON.stringify({
            prompt,
            files: files.map(f => f.serverPath),
            max_retries: parseInt(retriesSelect.value),
        }));
    };

    ws.onmessage = (e) => {
        const event = JSON.parse(e.data);
        handleEvent(event);
    };

    ws.onclose = () => finishSimulation();
    ws.onerror = () => finishSimulation();
}

function finishSimulation() {
    isRunning = false;
    
    // 恢复按钮状态至 Send
    actionBtn.classList.remove('stop-active');
    iconStop.style.display = 'none';
    iconSend.style.display = 'block';

    if (currentAssistantMsgInfo && currentAssistantMsgInfo.thinkIcon) {
        currentAssistantMsgInfo.thinkIcon.className = 'think-icon'; // 停止旋转
        currentAssistantMsgInfo.thinkIcon.textContent = '✔️';
    }
    // 把思考面板折叠
    if (currentAssistantMsgInfo && currentAssistantMsgInfo.thinkingPanel) {
        currentAssistantMsgInfo.thinkingPanel.classList.remove('open');
    }
    ws = null;
    scrollToBottom();
    // 任务结束后更新一下历史列表
    loadHistory();
}

// ── UI 渲染控制 ──────────────────────────────────────
function appendUserMessage(text, files) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    
    let filesHtml = '';
    if (files && files.length > 0) {
        filesHtml = '<div class="msg-files">';
        files.forEach(f => {
            const ext = f.name.split('.').pop().toLowerCase();
            if (['png','jpg','jpeg','gif','webp'].includes(ext) && f.serverPath) {
                // 如果是包含实际链接的图片
                const pathParts = f.serverPath.split(/[\\/]/);
                const sessionDirIndex = pathParts.findIndex(p => p.startsWith('session_'));
                let webUrl = '';
                if (sessionDirIndex !== -1) {
                    webUrl = `/files/uploads/${pathParts[sessionDirIndex]}/${pathParts[pathParts.length-1]}`;
                } else {
                    webUrl = `/files/uploads/${pathParts[pathParts.length-1]}`;
                }
                filesHtml += `<div class="msg-image-chip">
                                 <a href="${webUrl}" target="_blank" title="点击查看大图">
                                    <img src="${webUrl}" alt="${f.name}">
                                 </a>
                              </div>`;
            } else if (['png','jpg','jpeg','gif','webp'].includes(ext)) {
                 // 新发布时可能是临时上传的图片文件（在当前浏览器域内有内存指向）但还没拼好 WebUrl，由于我们并不想在这里做 FileReader 临时预览，只展出名字
                filesHtml += `<div class="msg-file-chip">🖼️ <span>${f.name}</span></div>`;
            } else {
                const icon = { pdf:'📕', docx:'📘', xlsx:'📊', pptx:'📙' }[ext] || '📄';
                filesHtml += `<div class="msg-file-chip">${icon} <span>${f.name}</span></div>`;
            }
        });
        filesHtml += '</div>';
    }

    const textualText = text ? `<div>${escapeHtml(text).replace(/\\n/g, '<br>')}</div>` : '';

    row.innerHTML = `<div class="message-bubble">${filesHtml}${textualText}</div>`;
    chatOutput.appendChild(row);
    scrollToBottom();
}

function prepareAssistantMessage() {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    row.innerHTML = `
        <div class="message-bubble">
            <div class="thinking-panel open" id="think-${Date.now()}">
                <div class="thinking-header">
                    <span class="think-icon running">⚙️</span>
                    <span>思考过程...</span>
                </div>
                <div class="thinking-content"></div>
            </div>
            <div class="assistant-response markdown-body"></div>
        </div>
    `;

    chatOutput.appendChild(row);

    // 绑定展开/折叠
    const panel = row.querySelector('.thinking-panel');
    row.querySelector('.thinking-header').addEventListener('click', () => {
        panel.classList.toggle('open');
    });

    currentAssistantMsgInfo = {
        row: row,
        thinkingPanel: panel,
        thinkContent: row.querySelector('.thinking-content'),
        thinkIcon: row.querySelector('.think-icon'),
        responseBody: row.querySelector('.assistant-response'),
        rawReportText: "" // 用于累加步骤 5 的 markdown 报告流
    };
    scrollToBottom();
}

function appendToThinking(html) {
    if (!currentAssistantMsgInfo.thinkContent) return;
    const div = document.createElement('div');
    div.className = 'log-line';
    div.innerHTML = html;
    currentAssistantMsgInfo.thinkContent.appendChild(div);
    currentAssistantMsgInfo.thinkContent.scrollTop = currentAssistantMsgInfo.thinkContent.scrollHeight;
    scrollToBottom();
}

function appendStreamToThinking(text) {
    // 简单地追加文本节点到末尾
    if (!currentAssistantMsgInfo.thinkContent) return;
    let lastLog = currentAssistantMsgInfo.thinkContent.lastElementChild;
    if (!lastLog || !lastLog.classList.contains('stream-text')) {
        lastLog = document.createElement('div');
        lastLog.className = 'log-line stream-text';
        currentAssistantMsgInfo.thinkContent.appendChild(lastLog);
    }
    lastLog.textContent += text;
    currentAssistantMsgInfo.thinkContent.scrollTop = currentAssistantMsgInfo.thinkContent.scrollHeight;
    scrollToBottom();
}

const STEP_NAMES = {
    1: '输入清晰化',
    2: '结构化解析',
    3: '代码构建',
    4: '执行与校正',
    5: '分析报告',
};

// ── 事件处理映射 ──────────────────────────────────────
function handleEvent(event) {
    const t = event.type;
    const step = event.step || 0;
    
    // 如果是报告步骤的流，将其输出到 markdown-body
    if (step === 5) {
        if (t === 'stream') {
            currentAssistantMsgInfo.rawReportText += event.chunk;
            // 增量实时渲染 markdown
            currentAssistantMsgInfo.responseBody.innerHTML = marked.parse(currentAssistantMsgInfo.rawReportText);
            scrollToBottom();
            return;
        }
                if (t === 'step_complete') {
            // 渲染并应用高亮
            currentAssistantMsgInfo.responseBody.innerHTML = marked.parse(currentAssistantMsgInfo.rawReportText);
            currentAssistantMsgInfo.responseBody.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
            // 触发 MathJax 渲染
            if (window.MathJax) {
                MathJax.typesetPromise([currentAssistantMsgInfo.responseBody]).catch(err => console.error(err));
            }
            scrollToBottom();
            return;
        }
    }

    // 其他步骤记录到 Thinking 面板中
    switch (t) {
        case 'step_start':
            appendToThinking(`<b>[开始]</b> 正在执行任务: ${STEP_NAMES[step] || '步骤 '+step}...`);
            break;
        case 'stream':
            appendStreamToThinking(event.chunk);
            break;
        case 'step_complete':
            appendToThinking(`<span class="success"><b>[完成]</b> ${STEP_NAMES[step] || '步骤 '+step} 已结束。</span>`);
            break;
        case 'step_error':
            appendToThinking(`<span class="error"><b>[错误]</b> ${escapeHtml(event.message)}</span>`);
            break;
        case 'attempt_start':
            appendToThinking(`<i>[重试] 第 ${event.attempt}/${event.max_retries} 次尝试...</i>`);
            break;
        case 'execution_result':
            const icon = { success: '✅', error: '❌', timeout: '⏰' }[event.status] || '❓';
            appendToThinking(`<i>执行结果: ${icon} ${event.status}</i>`);
            break;
        case 'info':
            appendToThinking(`[INFO] ${escapeHtml(event.message)}`);
            break;
        case 'image':
            // 图片属于结果的一部分，直接放入主回答
            if (currentAssistantMsgInfo.responseBody) {
                currentAssistantMsgInfo.responseBody.innerHTML += `<div style="margin-top:12px;"><img src="${event.path}" style="max-width:100%;border-radius:8px;border:1px solid #ddd;" /></div>`;
            }
            scrollToBottom();
            break;
        case 'pipeline_error':
            appendToThinking(`<span class="error"><b>[工作流崩溃/中止]</b> ${escapeHtml(event.message)}</span>`);
            if (currentAssistantMsgInfo.responseBody && currentAssistantMsgInfo.rawReportText.trim() === '') {
                currentAssistantMsgInfo.responseBody.innerHTML = `<span style="color:#D93025; font-weight:500;">任务失败/终止：${escapeHtml(event.message)}</span>`;
            }
            break;
        case 'pipeline_complete':
            // 结束标志，稍后 ws 会 close
            break;
    }
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatOutput.scrollTop = chatOutput.scrollHeight;
    });
}
