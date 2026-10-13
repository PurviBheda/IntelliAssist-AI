import { FormEvent, ReactNode, useState } from "react";

type IconName =
  | "arrow"
  | "chevron"
  | "clock"
  | "compass"
  | "file"
  | "grid"
  | "home"
  | "menu"
  | "mic"
  | "search"
  | "share"
  | "sparkles"
  | "star"
  | "templates";

const paths: Record<IconName, ReactNode> = {
  arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
  chevron: <path d="m8 10 4 4 4-4" />,
  clock: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 8v4l2.5 1.5" />
    </>
  ),
  compass: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="m15 9-2 4-4 2 2-4 4-2Z" />
    </>
  ),
  file: (
    <>
      <path d="M7 3h7l4 4v14H7z" />
      <path d="M14 3v5h5" />
    </>
  ),
  grid: (
    <>
      <rect x="4" y="4" width="6" height="6" rx="1" />
      <rect x="14" y="4" width="6" height="6" rx="1" />
      <rect x="4" y="14" width="6" height="6" rx="1" />
      <rect x="14" y="14" width="6" height="6" rx="1" />
    </>
  ),
  home: (
    <>
      <path d="m4 11 8-7 8 7" />
      <path d="M6 10v10h12V10M10 20v-6h4v6" />
    </>
  ),
  menu: (
    <>
      <path d="M5 7h14M5 12h14M5 17h14" />
    </>
  ),
  mic: (
    <>
      <rect x="9" y="3" width="6" height="12" rx="3" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
    </>
  ),
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="m16 16 4 4" />
    </>
  ),
  share: (
    <>
      <circle cx="18" cy="5" r="2" />
      <circle cx="6" cy="12" r="2" />
      <circle cx="18" cy="19" r="2" />
      <path d="m8 11 8-5M8 13l8 5" />
    </>
  ),
  sparkles: (
    <>
      <path d="m12 3 1.1 3.4L16.5 8l-3.4 1.3L12 13l-1.1-3.7L7.5 8l3.4-1.6L12 3Z" />
      <path d="m18.5 13 .7 2.1 2.1.9-2.1.8-.7 2.2-.7-2.2-2.1-.8 2.1-.9.7-2.1Z" />
    </>
  ),
  star: (
    <path d="m12 3 2.7 5.5 6.1.9-4.4 4.3 1 6.1-5.4-2.9-5.4 2.9 1-6.1-4.4-4.3 6.1-.9L12 3Z" />
  ),
  templates: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="3" />
      <path d="M9 4v16M9 10h11" />
    </>
  ),
};

function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      <g
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      >
        {paths[name]}
      </g>
    </svg>
  );
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  );
}

const projects = [
  ["clock", "Recent"],
  ["grid", "All Projects"],
  ["star", "Starred"],
  ["share", "Shared with me"],
] as const;

const resources = [
  ["compass", "Discover"],
  ["templates", "Templates"],
] as const;

export default function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [message, setMessage] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();

    if (!message.trim()) return;

    setMessage("");
  }

  return (
    <div className="app-shell">
      <header className="mobile-header">
        <button
          aria-label="Open navigation"
          onClick={() => setMobileOpen(true)}
        >
          <Icon name="menu" size={20} />
        </button>

        <div className="mobile-brand">
          <BrandMark />
          <span>IntelliAssist AI</span>
        </div>

        <div className="avatar">AK</div>
      </header>

      {mobileOpen && (
        <button
          aria-label="Close navigation"
          className="sidebar-backdrop"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <button className="workspace-button">
          <span className="workspace-icon">
            <Icon name="sparkles" size={13} />
          </span>

          <span>IntelliAssist AI</span>

          <Icon name="chevron" size={14} />
        </button>

        <nav aria-label="Primary navigation">
          <a className="nav-item active" href="#home">
            <Icon name="home" size={14} />
            Home
          </a>

          <a className="nav-item" href="#search">
            <Icon name="search" size={14} />
            Search
          </a>

          <p className="nav-label">Projects</p>

          {projects.map(([icon, label]) => (
            <a
              className="nav-item"
              href={`#${label.toLowerCase().replaceAll(" ", "-")}`}
              key={label}
            >
              <Icon name={icon} size={14} />
              {label}
            </a>
          ))}

          <p className="nav-label">Resources</p>

          {resources.map(([icon, label]) => (
            <a
              className="nav-item"
              href={`#${label.toLowerCase()}`}
              key={label}
            >
              <Icon name={icon} size={14} />
              {label}
            </a>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="profile-row">
            <span className="avatar">AK</span>

            <span className="profile-copy">
              <strong>Alex Kim</strong>
              <small>Free plan</small>
            </span>

            <span className="status-dot" />
          </button>
        </div>
      </aside>

      <main className="workspace" id="home">
        <div className="aurora" />

        <section className="prompt-section">
          <div className="announcement">
            <span>New</span>
            Meet your smarter document assistant
            <Icon name="arrow" size={13} />
          </div>

          <h1>What can I help you with today?</h1>

          <form className="composer" onSubmit={submit}>
            <label className="sr-only" htmlFor="message">
              Ask IntelliAssist AI
            </label>

            <textarea
              id="message"
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Ask anything about your documents"
              rows={1}
              value={message}
            />

            <div className="composer-actions">
              <div className="left-actions">
                <button
                  aria-label="Upload a document"
                  className="icon-button"
                  type="button"
                >
                  <span>+</span>
                </button>
              </div>

              <div className="right-actions">
                <button className="plan-button" type="button">
                  Plan
                </button>

                <button
                  aria-label="Use voice input"
                  className="icon-button"
                  type="button"
                >
                  <Icon name="mic" size={17} />
                </button>

                <button
                  aria-label="Send message"
                  className="send-button"
                  type="submit"
                >
                  <Icon name="arrow" size={17} />
                </button>
              </div>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}
