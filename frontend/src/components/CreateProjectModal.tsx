import { useState } from "react";

export default function CreateProjectModal({
  busy,
  onCreate,
  onClose,
}: {
  busy: boolean;
  onCreate: (name: string) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState("");

  return (
    <div className="modal-layer" role="dialog" aria-modal="true" aria-labelledby="create-project-title">
      <button type="button" className="modal-backdrop" onClick={onClose} aria-label="关闭新建项目" />
      <section className="create-project-modal">
        <header>
          <div><span>PROJECT</span><h2 id="create-project-title">新建项目</h2></div>
          <button type="button" onClick={onClose} aria-label="关闭">×</button>
        </header>
        <p>项目拥有持久工作区，可以上传资料，并集中下载 Agent 生成的文件。</p>
        <label>
          项目名称
          <input autoFocus maxLength={80} value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：发布说明" />
        </label>
        <footer>
          <button type="button" onClick={onClose}>取消</button>
          <button type="button" className="primary-button" disabled={!name.trim() || busy} onClick={() => onCreate(name.trim())}>
            {busy ? "创建中…" : "创建项目"}
          </button>
        </footer>
      </section>
    </div>
  );
}
