import {
  apiRequest,
  clearSession,
  clockLabel,
  requireSession,
  setButtonBusy,
  setStatus,
} from "/static/shared.js";

const session = requireSession();

if (session) {
  const profileButton = document.getElementById("profile-button");
  const logoutButton = document.getElementById("logout-button");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatThread = document.getElementById("chat-thread");
  const chatStatus = document.getElementById("chat-status");
  const healthChip = document.getElementById("chat-health");
  const sendButton = document.getElementById("send-button");
  const greeting = document.getElementById("chat-greeting");
  const promptButtons = [...document.querySelectorAll("[data-prompt]")];

  const userName = session.user.username || "there";
  const petName = session.user.pet_name ? ` for ${session.user.pet_name}` : "";

  greeting.textContent = `Hello ${userName}, ask anything${petName}.`;
  profileButton.textContent = userName;

  function appendMessage(role, content, options = {}) {
    const wrapper = document.createElement("article");
    wrapper.className = `message message-${role}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = `${role === "assistant" ? "Pet Doctor" : userName} • ${clockLabel()}`;

    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = content;

    if (options.pending) {
      body.classList.add("is-pending");
    }

    wrapper.append(meta, body);
    chatThread.append(wrapper);
    chatThread.scrollTop = chatThread.scrollHeight;

    return body;
  }

  async function checkHealth() {
    try {
      await apiRequest("/health");
      healthChip.textContent = "API Ready";
      healthChip.dataset.tone = "success";
    } catch (error) {
      healthChip.textContent = "API Down";
      healthChip.dataset.tone = "error";
      setStatus(chatStatus, error.message, "error");
    }
  }

  appendMessage(
    "assistant",
    `Welcome to the consultation window. I will answer using the backend service and available pet-health context.`
  );

  promptButtons.forEach((button) => {
    button.addEventListener("click", () => {
      chatInput.value = button.textContent.trim();
      chatInput.focus();
    });
  });

  profileButton.addEventListener("click", () => {
    window.location.href = "/profile";
  });

  logoutButton.addEventListener("click", () => {
    clearSession();
    window.location.href = "/";
  });

  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = chatInput.value.trim();

    if (!question) {
      setStatus(chatStatus, "Please enter a question first.", "error");
      return;
    }

    appendMessage("user", question);
    chatInput.value = "";
    const assistantBody = appendMessage("assistant", "Thinking...", { pending: true });

    setButtonBusy(sendButton, true, "Sending...");
    setStatus(chatStatus, "Waiting for backend response.", "neutral");

    try {
      const data = await apiRequest("/ask", {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      assistantBody.textContent = data.answer || "No answer returned.";
      assistantBody.classList.remove("is-pending");
      setStatus(chatStatus, "Answer received successfully.", "success");
    } catch (error) {
      assistantBody.textContent = error.message;
      assistantBody.classList.remove("is-pending");
      setStatus(chatStatus, "The backend request failed.", "error");
    } finally {
      setButtonBusy(sendButton, false, "Sending...");
    }
  });

  checkHealth();
}
