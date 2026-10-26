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
  fetchSharedConversation,
  removeDocument,
  sendChatMessage,
  shareConversation,
  toggleStarConversation,
  uploadDocument,
} from "./api"

type IconName = "arrow" | "check" | "chevron" | "clock" | "compass" | "dots" | "file" | "grid" | "home" | "menu" | "mic" | "plus" | "search" | "share" | "sparkles" | "star" | "templates" | "trash" | "upload" | "x"

const paths: Record<IconName, ReactNode> = {
  dots: (
    <>
      <circle cx="12" cy="5" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
      <circle cx="12" cy="19" r="1.5" fill="currentColor" stroke="none" />
    </>
  ),
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

type IconProps = {
  name: IconName
  size?: number
}

function Icon({ name, size = 16 }: IconProps) {
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

function formatConversationTitle(rawTitle: string): string {
  if (!rawTitle) return "Conversation"
  return (
    rawTitle
      .replace(/^[\s📄📑📁⭐*•\-]+\s*/gu, "")
      .replace(/\s*⭐+\s*$/gu, "")
      .trim() || rawTitle
  )
}

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
  const [starredExpanded, setStarredExpanded] = useState(true)
  const [sharedExpanded, setSharedExpanded] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchActive, setSearchActive] = useState(false)

  // Three-dot dropdown menu state
  const [menuTarget, setMenuTarget] = useState<{
    conversation: ConversationItem
    coords: {
      top: number
      right: number
    }
  } | null>(null)

  // Share Dialog state
  const [shareDialogConv, setShareDialogConv] =
    useState<ConversationItem | null>(null)
  const [shareLink, setShareLink] = useState("")
  const [copiedLink, setCopiedLink] = useState(false)
  const [isSharing, setIsSharing] = useState(false)

  // Delete Confirmation Dialog state
  const [deleteConfirmConv, setDeleteConfirmConv] =
    useState<ConversationItem | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)

  // Shared View Mode (when opened via share link)
  const [isSharedView, setIsSharedView] = useState(false)

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

  // Check URL params for share identifier on mount
  useEffect(() => {
    async function checkShareParam() {
      const urlParams = new URLSearchParams(window.location.search)
      const shareId = urlParams.get("share")
      if (shareId) {
        try {
          showToast("Loading shared conversation...", "info")
          const sharedConv = await fetchSharedConversation(shareId)
          setActiveConversationId(sharedConv.conversation_id)
          setActiveConversationTitle(sharedConv.title || "Shared Conversation")
          setMessages(sharedConv.messages || [])
          if (sharedConv.documents && sharedConv.documents.length > 0) {
            setCurrentDocument(
              sharedConv.documents[sharedConv.documents.length - 1],
            )
          } else {
            setCurrentDocument(null)
          }
          setIsSharedView(true)
          showToast("Loaded shared conversation", "success")
        } catch (err: any) {
          showToast(
            err.message || "Shared conversation not found or access revoked",
            "error",
          )
        }
      }
    }
    checkShareParam()
  }, [])

  // Dismiss menu and modals on outside click, scroll, or Escape key
  useEffect(() => {
    function handleGlobalClose(e: MouseEvent | KeyboardEvent) {
      if (e instanceof KeyboardEvent && e.key === "Escape") {
        setMenuTarget(null)
        setShareDialogConv(null)
        setDeleteConfirmConv(null)
        return
      }
      setMenuTarget(null)
    }

    window.addEventListener("mousedown", handleGlobalClose)
    window.addEventListener("scroll", handleGlobalClose, true)
    window.addEventListener("keydown", handleGlobalClose)
    return () => {
      window.removeEventListener("mousedown", handleGlobalClose)
      window.removeEventListener("scroll", handleGlobalClose, true)
      window.removeEventListener("keydown", handleGlobalClose)
    }
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
    setIsSharedView(false)
    setMobileOpen(false)
  }

  async function openConversation(conv: ConversationItem) {
    try {
      setActiveConversationId(conv.conversation_id)
      setActiveConversationTitle(conv.title || "Conversation")
      setIsSharedView(false)
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

  function handleOpenMenu(event: React.MouseEvent, conv: ConversationItem) {
    event.stopPropagation()
    const target = event.currentTarget as HTMLElement
    const rect = target.getBoundingClientRect()
    const right = Math.max(8, window.innerWidth - rect.right)
    const top = rect.bottom + 4
    setMenuTarget({
      conversation: conv,
      coords: { top, right },
    })
  }

  async function handleToggleStar(conv: ConversationItem) {
    setMenuTarget(null)
    const targetState = !conv.is_starred

    // Optimistically update conversations state
    setConversations((prev) =>
      prev.map((c) =>
        c.conversation_id === conv.conversation_id
          ? { ...c, is_starred: targetState }
          : c,
      ),
    )

    try {
      await toggleStarConversation(conv.conversation_id, targetState)
      showToast(
        targetState ? "Conversation starred" : "Conversation unstarred",
        "success",
      )
    } catch (err: any) {
      // Revert on error
      setConversations((prev) =>
        prev.map((c) =>
          c.conversation_id === conv.conversation_id
            ? { ...c, is_starred: !targetState }
            : c,
        ),
      )
      showToast(err.message || "Failed to update star status", "error")
    }
  }

  async function handleOpenShareDialog(conv: ConversationItem) {
    setMenuTarget(null)
    setShareDialogConv(conv)
    setCopiedLink(false)

    try {
      setIsSharing(true)
      const res = await shareConversation(conv.conversation_id)
      const sid = res.share_id
      setConversations((prev) =>
        prev.map((c) =>
          c.conversation_id === conv.conversation_id
            ? { ...c, is_shared: true, share_id: sid }
            : c,
        ),
      )
      const url = `${window.location.origin}/?share=${sid}`
      setShareLink(url)
    } catch (err: any) {
      showToast(err.message || "Failed to generate share link", "error")
    } finally {
      setIsSharing(false)
    }
  }

  function handleCopyShareLink() {
    if (!shareLink) return
    navigator.clipboard
      .writeText(shareLink)
      .then(() => {
        setCopiedLink(true)
        showToast("Link copied to clipboard", "success")
        setTimeout(() => setCopiedLink(false), 2500)
      })
      .catch(() => {
        showToast("Failed to copy link to clipboard", "error")
      })
  }

  function handleOpenDeleteDialog(conv: ConversationItem) {
    setMenuTarget(null)
    setDeleteConfirmConv(conv)
  }

  async function handleConfirmDelete() {
    if (!deleteConfirmConv) return
    const cid = deleteConfirmConv.conversation_id
    setIsDeleting(true)
    try {
      await deleteConversation(cid)
      setConversations((prev) => prev.filter((c) => c.conversation_id !== cid))
      if (activeConversationId === cid) {
        startNewChat()
      }
      setDeleteConfirmConv(null)
      showToast("Conversation deleted", "success")
    } catch (err: any) {
      showToast(err.message || "Failed to delete conversation", "error")
    } finally {
      setIsDeleting(false)
    }
  }

  function renderConversationItem(c: ConversationItem) {
    const isActive = activeConversationId === c.conversation_id
    const isMenuOpen =
      menuTarget?.conversation.conversation_id === c.conversation_id
    const displayTitle = formatConversationTitle(c.title)

    return (
      <div
        className={`recent-item ${isActive ? "active" : ""}`}
        key={c.conversation_id}
        onClick={() => {
          setIsSharedView(false)
          openConversation(c)
        }}
        title={displayTitle}
      >
        <div className="recent-item-title">
          <span>{displayTitle}</span>
        </div>

        <div
          className="recent-item-actions"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            aria-expanded={isMenuOpen}
            aria-haspopup="true"
            aria-label={`Options for ${displayTitle}`}
            className={`three-dots-btn ${isMenuOpen ? "menu-open" : ""}`}
            onClick={(e) => handleOpenMenu(e, c)}
            title="More options"
            type="button"
          >
            <Icon name="dots" size={13} />
          </button>
        </div>
      </div>
    )
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

  const starredConversations = conversations.filter((c) => !!c.is_starred)
  const sharedConversations = conversations.filter((c) => !!c.is_shared)

  const filteredRecent = searchQuery.trim()
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase().trim()),
      )
    : conversations

  const filteredStarred = searchQuery.trim()
    ? starredConversations.filter((c) =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase().trim()),
      )
    : starredConversations

  const filteredShared = searchQuery.trim()
    ? sharedConversations.filter((c) =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase().trim()),
      )
    : sharedConversations

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

        <div className="avatar">PB</div>
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
                    fontSize: "12px",
                    padding: "4px 8px",
                    outline: "none",
                  }}
                  value={searchQuery}
                />
              </div>
            )}
          </div>

          <p className="nav-label">Projects</p>

          {/* 1. Recent Section */}
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
              {filteredRecent.length === 0 ? (
                <div className="recent-empty-label">
                  {searchQuery
                    ? "No matching conversations"
                    : "No recent chats yet"}
                </div>
              ) : (
                filteredRecent.map(renderConversationItem)
              )}
            </div>
          )}

          {/* 2. Starred Section */}
          <button
            className="nav-item"
            onClick={() => setStarredExpanded(!starredExpanded)}
            style={{ width: "100%", justifyContent: "space-between" }}
          >
            <span
              style={{ display: "flex", alignItems: "center", gap: "10px" }}
            >
              <Icon name="star" size={14} />
              Starred
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              {starredConversations.length > 0 && (
                <span className="recent-badge-count">
                  {starredConversations.length}
                </span>
              )}
              <Icon name="chevron" size={12} />
            </span>
          </button>

          {starredExpanded && (
            <div className="recent-list">
              {filteredStarred.length === 0 ? (
                <div className="recent-empty-label">
                  No starred conversations yet.
                  <br />
                  Star a conversation to find it here.
                </div>
              ) : (
                filteredStarred.map(renderConversationItem)
              )}
            </div>
          )}

          {/* 3. Shared with me Section */}
          <button
            className="nav-item"
            onClick={() => setSharedExpanded(!sharedExpanded)}
            style={{ width: "100%", justifyContent: "space-between" }}
          >
            <span
              style={{ display: "flex", alignItems: "center", gap: "10px" }}
            >
              <Icon name="share" size={14} />
              Shared with me
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              {sharedConversations.length > 0 && (
                <span className="recent-badge-count">
                  {sharedConversations.length}
                </span>
              )}
              <Icon name="chevron" size={12} />
            </span>
          </button>

          {sharedExpanded && (
            <div className="recent-list">
              {filteredShared.length === 0 ? (
                <div className="recent-empty-label">
                  No shared conversations yet.
                  <br />
                  Share a conversation to see it here.
                </div>
              ) : (
                filteredShared.map(renderConversationItem)
              )}
            </div>
          )}
        </nav>

        <div className="sidebar-footer">
          <button className="profile-row">
            <span className="avatar">PB</span>

            <span className="profile-copy">
              <strong>Purvi Bheda</strong>
              <small>Free plan</small>
            </span>

            <span className="status-dot" />
          </button>
        </div>
      </aside>

      <main className="workspace" id="home">
        <div className="aurora" />

        {/* Read-only shared view top banner */}
        {isSharedView && (
          <div className="shared-view-banner">
            <span>
              Viewing shared conversation:{" "}
              <strong>{activeConversationTitle}</strong> (Read-only)
            </span>
            <button
              className="btn-secondary"
              onClick={() => {
                setIsSharedView(false)
                startNewChat()
                window.history.replaceState(
                  {},
                  document.title,
                  window.location.pathname,
                )
              }}
              style={{ padding: "3px 8px", fontSize: "12px" }}
              type="button"
            >
              Back to my chats
            </button>
          </div>
        )}

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

      {/* Floating Three-dot Dropdown Menu */}
      {menuTarget && (
        <div
          aria-label="Conversation options"
          className="convo-dropdown-menu"
          onMouseDown={(e) => e.stopPropagation()}
          role="menu"
          style={{
            top: menuTarget.coords.top,
            right: menuTarget.coords.right,
          }}
        >
          <button
            className="dropdown-item"
            onClick={() => handleToggleStar(menuTarget.conversation)}
            role="menuitem"
            type="button"
          >
            <Icon name="star" size={13} />
            <span>
              {menuTarget.conversation.is_starred ? "Unstar" : "Star"}
            </span>
          </button>

          <button
            className="dropdown-item"
            onClick={() => handleOpenShareDialog(menuTarget.conversation)}
            role="menuitem"
            type="button"
          >
            <Icon name="share" size={13} />
            <span>Share</span>
          </button>

          <button
            className="dropdown-item danger"
            onClick={() => handleOpenDeleteDialog(menuTarget.conversation)}
            role="menuitem"
            type="button"
          >
            <Icon name="trash" size={13} />
            <span>Delete</span>
          </button>
        </div>
      )}

      {/* Share Conversation Modal Dialog */}
      {shareDialogConv && (
        <div
          aria-labelledby="share-dialog-title"
          aria-modal="true"
          className="modal-overlay"
          onClick={() => setShareDialogConv(null)}
          role="dialog"
        >
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title" id="share-dialog-title">
                Share conversation
              </h2>
              <button
                aria-label="Close"
                className="modal-close-btn"
                onClick={() => setShareDialogConv(null)}
                type="button"
              >
                <Icon name="x" size={14} />
              </button>
            </div>

            <div className="share-conv-title">{shareDialogConv.title}</div>

            <p
              style={{
                fontSize: "12px",
                color: "#9ca3af",
                marginBottom: "12px",
                lineHeight: "1.4",
              }}
            >
              Anyone with this link can view this conversation in read-only
              mode.
            </p>

            <div className="share-link-box">
              <input
                aria-label="Shareable link"
                className="share-link-input"
                readOnly
                value={shareLink || "Generating link..."}
              />
              <button
                className={`share-copy-btn ${copiedLink ? "copied" : ""}`}
                disabled={!shareLink || isSharing}
                onClick={handleCopyShareLink}
                type="button"
              >
                {copiedLink ? (
                  <>
                    <Icon name="check" size={13} />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Icon name="share" size={13} />
                    <span>Copy link</span>
                  </>
                )}
              </button>
            </div>

            <div className="modal-footer">
              <button
                className="btn-secondary"
                onClick={() => setShareDialogConv(null)}
                type="button"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal Dialog */}
      {deleteConfirmConv && (
        <div
          aria-labelledby="delete-dialog-title"
          aria-modal="true"
          className="modal-overlay"
          onClick={() => !isDeleting && setDeleteConfirmConv(null)}
          role="dialog"
        >
          <div
            className="modal-dialog"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 390 }}
          >
            <div className="modal-header">
              <h2 className="modal-title" id="delete-dialog-title">
                Delete conversation?
              </h2>
              <button
                aria-label="Close"
                className="modal-close-btn"
                disabled={isDeleting}
                onClick={() => setDeleteConfirmConv(null)}
                type="button"
              >
                <Icon name="x" size={14} />
              </button>
            </div>

            <p
              style={{
                fontSize: "13.5px",
                color: "#a1a1aa",
                margin: "4px 0 16px",
                lineHeight: "1.5",
              }}
            >
              This conversation and its chat history will be deleted.
            </p>

            <div className="modal-footer">
              <button
                className="btn-secondary"
                disabled={isDeleting}
                onClick={() => setDeleteConfirmConv(null)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="btn-danger"
                disabled={isDeleting}
                onClick={handleConfirmDelete}
                type="button"
              >
                {isDeleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
