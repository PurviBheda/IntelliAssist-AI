import { FormEvent, ReactNode, useEffect, useRef, useState } from "react"
import {
  ChatMessage,
  Citation,
  ConversationItem,
  DocumentRecord,
  createConversation,
  deleteConversation,
  fetchConversationDetails,
  fetchConversations,
  removeDocument,
  sendChatMessage,
  uploadDocument,
} from "./api"

type IconName = "arrow" | "check" | "chevron" | "clock" | "compass" | "file" | "grid" | "home" | "menu" | "mic" | "plus" | "search" | "share" | "sparkles" | "star" | "templates" | "trash" | "upload" | "x"

const paths: Record<IconName, ReactNode> = {
  arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
  check: <path d="m20 6-11 11-5-5" />,
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
  plus: <path d="M12 5v14M5 12h14" />,
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
  trash: (
    <>
      <path d="M3 6h18" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </>
  ),
  upload: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="m17 8-5-5-5 5" />
      <path d="M12 3v12" />
    </>
  ),
  x: <path d="m18 6-12 12M6 6l12 12" />,
}

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
  )
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}

const projects = [
  ["clock", "Recent"],
  ["grid", "All Projects"],
  ["star", "Starred"],
  ["share", "Shared with me"],
] as const

const resources = [
  ["compass", "Discover"],
  ["templates", "Templates"],
] as const

export default function App() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [message, setMessage] = useState("")

  // Conversation & History State
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [activeConversationId, setActiveConversationId] =
    useState<string | null>(null)
  const [activeConversationTitle, setActiveConversationTitle] =
    useState("New Conversation")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [currentDocument, setCurrentDocument] = useState<DocumentRecord | null>(
    null,
  )

  // Upload & Async States
  const [uploadMenuOpen, setUploadMenuOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [recentExpanded, setRecentExpanded] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchActive, setSearchActive] = useState(false)

  // Notification Toast
  const [toast, setToast] = useState<{
    message: string
    type: "success" | "error" | "info"
  } | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const uploadMenuAnchorRef = useRef<HTMLDivElement>(null)

  // Auto-scroll messages to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isGenerating])

  // Load conversations list on mount
  useEffect(() => {
    loadConversations()
  }, [])

  // Dismiss upload menu on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        uploadMenuAnchorRef.current &&
        !uploadMenuAnchorRef.current.contains(event.target as Node)
      ) {
        setUploadMenuOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  // Toast auto-dismiss
  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => {
      setToast(null)
    }, 4500)
    return () => clearTimeout(timer)
  }, [toast])

  async function loadConversations() {
    try {
      const list = await fetchConversations(50)
      setConversations(list)
    } catch {
      // Backend may be initializing; fail gracefully
    }
  }

  function showToast(msg: string, type: "success" | "error" | "info" = "info") {
    setToast({ message: msg, type })
  }

  function startNewChat() {
    setActiveConversationId(null)
    setActiveConversationTitle("New Conversation")
    setMessages([])
    setCurrentDocument(null)
    setMobileOpen(false)
  }

  async function openConversation(conv: ConversationItem) {
    try {
      setActiveConversationId(conv.conversation_id)
      setActiveConversationTitle(conv.title || "Conversation")
      setMobileOpen(false)

      const details = await fetchConversationDetails(conv.conversation_id)
      setMessages(details.messages || [])

      if (details.documents && details.documents.length > 0) {
        setCurrentDocument(details.documents[details.documents.length - 1])
      } else {
        setCurrentDocument(null)
      }
    } catch (err: any) {
      showToast(err.message || "Failed to load conversation history", "error")
    }
  }

  async function handleDeleteConversation(
    event: React.MouseEvent,
    cid: string,
  ) {
    event.stopPropagation()
    try {
      await deleteConversation(cid)
      setConversations((prev) => prev.filter((c) => c.conversation_id !== cid))

      if (activeConversationId === cid) {
        startNewChat()
      }
      showToast("Conversation deleted", "info")
    } catch (err: any) {
      showToast(err.message || "Failed to delete conversation", "error")
    }
  }

  function triggerFilePicker() {
    setUploadMenuOpen(false)
    fileInputRef.current?.click()
  }

  async function handleFileSelected(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0]
    if (!file) return

    // Reset input value so same file can be re-selected if desired
    event.target.value = ""

    const name = file.name.toLowerCase()
    const isSupported =
      name.endsWith(".pdf") || name.endsWith(".docx") || name.endsWith(".txt")
    if (!isSupported) {
      showToast("Unsupported format. Please upload PDF, DOCX, or TXT.", "error")
      return
    }

    setIsUploading(true)
    showToast(`Uploading and processing "${file.name}"...`, "info")

    try {
      // If we don't have an active conversation, create one first or pass undefined
      let targetConvId = activeConversationId
      if (!targetConvId) {
        const newConv = await createConversation(`Doc: ${file.name}`)
        targetConvId = newConv.conversation_id
        setActiveConversationId(targetConvId)
        setActiveConversationTitle(newConv.title)
      }

      const res = await uploadDocument(file, targetConvId)
      setCurrentDocument(res.document)
      showToast(
        `Ready: "${res.document.filename}" is indexed for Q&A.`,
        "success",
      )
      await loadConversations()
    } catch (err: any) {
      showToast(
        err.message || "Document processing failed. Please check file format.",
        "error",
      )
    } finally {
      setIsUploading(false)
    }
  }

  async function handleRemoveDocument() {
    if (!currentDocument) return
    const docId = currentDocument.document_id
    const docName = currentDocument.filename

    if (activeConversationId && docId) {
      try {
        await removeDocument(activeConversationId, docId)
      } catch {
        // Silently continue locally
      }
    }

    setCurrentDocument(null)
    showToast(`Removed "${docName}" from current conversation.`, "info")
  }

  async function submit(event: FormEvent) {
    event.preventDefault()

    const trimmed = message.trim()
    if (!trimmed || isGenerating) return

    setMessage("")

    // Optimistically add user message
    const userMsg: ChatMessage = {
      role: "user",
      content: trimmed,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])
    setIsGenerating(true)

    try {
      let targetConvId = activeConversationId
      if (!targetConvId) {
        // Create conversation
        const newConv = await createConversation()
        targetConvId = newConv.conversation_id
        setActiveConversationId(targetConvId)
        setActiveConversationTitle(newConv.title)
      }

      const res = await sendChatMessage(targetConvId, trimmed)

      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: res.assistant_response,
        citations: res.citations || [],
        created_at: res.created_at,
        metadata: {
          is_live_llm: res.is_live_llm,
          model_name: res.model_name,
          sufficient_context: res.sufficient_context,
          intent: res.intent,
        },
      }

      setMessages((prev) => [...prev, assistantMsg])
      await loadConversations()
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: `Error: ${err.message || "Failed to generate answer. Please try again."}`,
        created_at: new Date().toISOString(),
        metadata: { status: "error" },
      }
      setMessages((prev) => [...prev, errorMsg])
      showToast(err.message || "Chat failed. Please try again.", "error")
    } finally {
      setIsGenerating(false)
    }
  }

  const filteredConversations = searchQuery.trim()
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase().trim()),
      )
    : conversations

  return (
    <div className="app-shell">
      {/* Hidden file input for document uploading */}
      <input
        accept=".pdf,.docx,.txt"
        aria-label="Upload document"
        className="sr-only"
        onChange={handleFileSelected}
        ref={fileInputRef}
        type="file"
      />

      {/* Floating Toast Notification */}
      {toast && (
        <div className={`app-toast ${toast.type}`} role="status">
          {toast.type === "success" && <Icon name="check" size={14} />}
          {toast.type === "error" && <Icon name="x" size={14} />}
          {toast.type === "info" && <Icon name="sparkles" size={14} />}
          <span>{toast.message}</span>
        </div>
      )}

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
        <button
          className="workspace-button"
          onClick={startNewChat}
          title="Start a new conversation"
        >
          <span className="workspace-icon">
            <Icon name="sparkles" size={13} />
          </span>

          <span>IntelliAssist AI</span>

          <Icon name="plus" size={14} />
        </button>

        <nav aria-label="Primary navigation">
          <button
            className={`nav-item ${
              activeConversationId === null && messages.length === 0
                ? "active"
                : ""
            }`}
            onClick={startNewChat}
            style={{ width: "100%", justifyContent: "flex-start" }}
          >
            <Icon name="home" size={14} />
            Home
          </button>

          <div style={{ position: "relative" }}>
            <button
              className={`nav-item ${searchActive ? "active" : ""}`}
              onClick={() => setSearchActive(!searchActive)}
              style={{ width: "100%", justifyContent: "flex-start" }}
            >
              <Icon name="search" size={14} />
              Search
            </button>

            {searchActive && (
              <div style={{ padding: "4px 8px 6px" }}>
                <input
                  autoFocus
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter conversations..."
                  style={{
                    width: "100%",
                    background: "#1c1c1f",
                    border: "1px solid #36363d",
                    borderRadius: "6px",
                    color: "#fff",
                    fontSize: "11px",
                    padding: "4px 8px",
                    outline: "none",
                  }}
                  value={searchQuery}
                />
              </div>
            )}
          </div>

          <p className="nav-label">Projects</p>

          {/* Functional Recent Section */}
          <button
            className="nav-item"
            onClick={() => setRecentExpanded(!recentExpanded)}
            style={{ width: "100%", justifyContent: "space-between" }}
          >
            <span
              style={{ display: "flex", alignItems: "center", gap: "10px" }}
            >
              <Icon name="clock" size={14} />
              Recent
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              {conversations.length > 0 && (
                <span className="recent-badge-count">
                  {conversations.length}
                </span>
              )}
              <Icon name="chevron" size={12} />
            </span>
          </button>

          {recentExpanded && (
            <div className="recent-list">
              {filteredConversations.length === 0 ? (
                <div className="recent-empty-label">
                  {searchQuery
                    ? "No matching conversations"
                    : "No recent chats yet"}
                </div>
              ) : (
                filteredConversations.map((c) => {
                  const isActive = activeConversationId === c.conversation_id
                  const hasDoc = c.documents && c.documents.length > 0
                  return (
                    <div
                      className={`recent-item ${isActive ? "active" : ""}`}
                      key={c.conversation_id}
                      onClick={() => openConversation(c)}
                      title={c.title}
                    >
                      <div className="recent-item-title">
                        <Icon name={hasDoc ? "file" : "sparkles"} size={12} />
                        <span>{c.title}</span>
                      </div>

                      <button
                        aria-label="Delete conversation"
                        className="recent-item-delete"
                        onClick={(e) =>
                          handleDeleteConversation(e, c.conversation_id)
                        }
                        title="Delete conversation"
                      >
                        <Icon name="trash" size={11} />
                      </button>
                    </div>
                  )
                })
              )}
            </div>
          )}

          {/* Remaining Projects Items */}
          {projects
            .filter(([_, label]) => label !== "Recent")
            .map(([icon, label]) => (
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

        {/* Mode A: Welcome Screen (When no messages in current thread) */}
        {messages.length === 0 ? (
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

              {/* Uploaded document badge above composer textarea */}
              {currentDocument && (
                <div className="composer-doc-badge">
                  <div className="composer-doc-left">
                    <span className="doc-icon">
                      <Icon name="file" size={14} />
                    </span>
                    <span className="doc-name">{currentDocument.filename}</span>
                    <span className="doc-meta">
                      ({currentDocument.total_pages}{" "}
                      {currentDocument.total_pages === 1 ? "page" : "pages"})
                    </span>
                    <span className="doc-status-ready">
                      <span
                        style={{
                          width: 6,
                          height: 6,
                          borderRadius: "50%",
                          background: "#34d399",
                          display: "inline-block",
                        }}
                      />
                      Ready for questions
                    </span>
                  </div>

                  <button
                    aria-label="Remove document"
                    className="doc-remove-btn"
                    onClick={handleRemoveDocument}
                    type="button"
                  >
                    <Icon name="x" size={13} />
                  </button>
                </div>
              )}

              {/* Uploading progress indicator */}
              {isUploading && (
                <div className="composer-doc-badge">
                  <div className="composer-doc-left">
                    <span className="spinner" />
                    <span className="doc-name">
                      Processing and indexing document...
                    </span>
                  </div>
                </div>
              )}

              <textarea
                id="message"
                onChange={(event) => setMessage(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault()
                    submit(event)
                  }
                }}
                placeholder={
                  currentDocument
                    ? `Ask anything about ${currentDocument.filename}...`
                    : "Ask anything about your documents"
                }
                rows={1}
                value={message}
              />

              <div className="composer-actions">
                <div className="left-actions">
                  <div
                    className="upload-popover-anchor"
                    ref={uploadMenuAnchorRef}
                  >
                    <button
                      aria-label="Upload a document"
                      className="icon-button"
                      onClick={() => setUploadMenuOpen(!uploadMenuOpen)}
                      type="button"
                    >
                      <span>+</span>
                    </button>

                    {/* Plus Button Popover Menu */}
                    {uploadMenuOpen && (
                      <div className="upload-popover">
                        <button
                          className="upload-menu-item"
                          onClick={triggerFilePicker}
                          type="button"
                        >
                          <div className="upload-menu-icon">
                            <Icon name="upload" size={15} />
                          </div>
                          <div className="upload-menu-text">
                            <span className="upload-menu-title">
                              Upload files
                            </span>
                            <span className="upload-menu-hint">
                              PDF, DOCX, or TXT documents
                            </span>
                          </div>
                        </button>
                      </div>
                    )}
                  </div>
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
                    disabled={isGenerating || !message.trim()}
                    style={{ opacity: message.trim() ? 1 : 0.6 }}
                    type="submit"
                  >
                    <Icon name="arrow" size={17} />
                  </button>
                </div>
              </div>
            </form>
          </section>
        ) : (
          /* Mode B: Active Conversation Screen */
          <div className="conversation-view">
            {/* Topbar */}
            <div className="conversation-topbar">
              <div className="conversation-meta">
                <span className="conversation-title-text">
                  {activeConversationTitle}
                </span>

                {currentDocument && (
                  <div className="conversation-doc-chip">
                    <Icon name="file" size={12} />
                    <span>{currentDocument.filename}</span>
                  </div>
                )}
              </div>

              <button
                className="new-chat-button"
                onClick={startNewChat}
                type="button"
              >
                <Icon name="plus" size={13} />
                New Chat
              </button>
            </div>

            {/* Scrollable Messages Stream */}
            <div className="conversation-messages">
              {messages.map((msg, index) => (
                <div
                  className={`message-row ${
                    msg.role === "user" ? "user-row" : "assistant-row"
                  }`}
                  key={index}
                >
                  {msg.role === "user" ? (
                    <div className="user-bubble">{msg.content}</div>
                  ) : (
                    <div className="assistant-bubble-container">
                      <div className="assistant-avatar">
                        <Icon name="sparkles" size={15} />
                      </div>

                      <div className="assistant-bubble">
                        {/* Model / Grounding Badge */}
                        <div className="model-tag">
                          <span>
                            {msg.metadata?.is_live_llm
                              ? "Gemini 3.6 Flash"
                              : "IntelliAssist Grounded RAG"}
                          </span>
                        </div>

                        {/* Message Content */}
                        <div className="assistant-text">{msg.content}</div>

                        {/* Insufficient context banner */}
                        {msg.metadata?.sufficient_context === false && (
                          <div className="insufficient-context-banner">
                            <Icon name="file" size={13} />
                            <span>
                              This answer was strictly grounded; unverified
                              facts were excluded.
                            </span>
                          </div>
                        )}

                        {/* Citations & Source References */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="citations-section">
                            <div className="citations-header">
                              <Icon name="file" size={12} />
                              <span>
                                Sources & References ({msg.citations.length})
                              </span>
                            </div>

                            <div className="citation-list">
                              {msg.citations.map(
                                (c: Citation, cIdx: number) => (
                                  <div
                                    className="citation-chip"
                                    key={cIdx}
                                    title={`Chunk: ${c.chunk_id || "n/a"}`}
                                  >
                                    <Icon name="file" size={11} />
                                    <span>{c.filename || "document"}</span>
                                    {c.page_number && (
                                      <span className="page-tag">
                                        p. {c.page_number}
                                      </span>
                                    )}
                                    {c.similarity_score !== undefined && (
                                      <span className="score-tag">
                                        {Math.round(c.similarity_score * 100)}%
                                      </span>
                                    )}
                                  </div>
                                ),
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {/* Generating loading indicator */}
              {isGenerating && (
                <div className="message-row assistant-row">
                  <div className="assistant-bubble-container">
                    <div className="assistant-avatar">
                      <Icon name="sparkles" size={15} />
                    </div>

                    <div
                      className="assistant-bubble"
                      style={{ padding: "12px 16px" }}
                    >
                      <div className="generating-indicator">
                        <div className="pulse-dots">
                          <span />
                          <span />
                          <span />
                        </div>
                        <span>
                          Searching documents and generating grounded answer...
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Bottom Composer in Conversation Mode */}
            <div className="conversation-bottom-bar">
              <form className="composer" onSubmit={submit}>
                {/* Uploaded document badge above composer textarea */}
                {currentDocument && (
                  <div className="composer-doc-badge">
                    <div className="composer-doc-left">
                      <span className="doc-icon">
                        <Icon name="file" size={14} />
                      </span>
                      <span className="doc-name">
                        {currentDocument.filename}
                      </span>
                      <span className="doc-meta">
                        ({currentDocument.total_pages}{" "}
                        {currentDocument.total_pages === 1 ? "page" : "pages"})
                      </span>
                      <span className="doc-status-ready">
                        <span
                          style={{
                            width: 6,
                            height: 6,
                            borderRadius: "50%",
                            background: "#34d399",
                            display: "inline-block",
                          }}
                        />
                        Ready for questions
                      </span>
                    </div>

                    <button
                      aria-label="Remove document"
                      className="doc-remove-btn"
                      onClick={handleRemoveDocument}
                      type="button"
                    >
                      <Icon name="x" size={13} />
                    </button>
                  </div>
                )}

                {/* Uploading progress indicator */}
                {isUploading && (
                  <div className="composer-doc-badge">
                    <div className="composer-doc-left">
                      <span className="spinner" />
                      <span className="doc-name">
                        Processing and indexing document...
                      </span>
                    </div>
                  </div>
                )}

                <textarea
                  id="conversation-message"
                  onChange={(event) => setMessage(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault()
                      submit(event)
                    }
                  }}
                  placeholder={
                    currentDocument
                      ? `Ask a follow-up about ${currentDocument.filename}...`
                      : "Ask anything about your documents"
                  }
                  rows={1}
                  value={message}
                />

                <div className="composer-actions">
                  <div className="left-actions">
                    <div
                      className="upload-popover-anchor"
                      ref={uploadMenuAnchorRef}
                    >
                      <button
                        aria-label="Upload a document"
                        className="icon-button"
                        onClick={() => setUploadMenuOpen(!uploadMenuOpen)}
                        type="button"
                      >
                        <span>+</span>
                      </button>

                      {uploadMenuOpen && (
                        <div className="upload-popover">
                          <button
                            className="upload-menu-item"
                            onClick={triggerFilePicker}
                            type="button"
                          >
                            <div className="upload-menu-icon">
                              <Icon name="upload" size={15} />
                            </div>
                            <div className="upload-menu-text">
                              <span className="upload-menu-title">
                                Upload files
                              </span>
                              <span className="upload-menu-hint">
                                PDF, DOCX, or TXT documents
                              </span>
                            </div>
                          </button>
                        </div>
                      )}
                    </div>
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
                      disabled={isGenerating || !message.trim()}
                      style={{ opacity: message.trim() ? 1 : 0.6 }}
                      type="submit"
                    >
                      <Icon name="arrow" size={17} />
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
