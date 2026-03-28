import {
  apiRequest,
  redirectIfAuthenticated,
  saveSession,
  serializeForm,
  setButtonBusy,
  setStatus,
} from "/static/shared.js";

if (!redirectIfAuthenticated("/chat")) {
  const authStatus = document.getElementById("auth-status");
  const loginForm = document.getElementById("login-form");
  const registerForm = document.getElementById("register-form");
  const loginSubmit = document.getElementById("login-submit");
  const registerSubmit = document.getElementById("register-submit");
  const tabButtons = [...document.querySelectorAll("[data-auth-tab]")];

  function setTab(mode) {
    tabButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.authTab === mode);
    });

    loginForm.classList.toggle("is-hidden", mode !== "login");
    registerForm.classList.toggle("is-hidden", mode !== "register");
  }

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.authTab));
  });

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = serializeForm(loginForm);

    setButtonBusy(loginSubmit, true, "Signing In...");
    setStatus(authStatus, "Checking your credentials.", "neutral");

    try {
      const data = await apiRequest("/login", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      saveSession(data.user);
      setStatus(authStatus, "Login successful. Opening workspace.", "success");
      window.location.href = "/chat";
    } catch (error) {
      setStatus(authStatus, error.message, "error");
    } finally {
      setButtonBusy(loginSubmit, false, "Signing In...");
    }
  });

  registerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = serializeForm(registerForm);

    setButtonBusy(registerSubmit, true, "Creating...");
    setStatus(authStatus, "Creating your account.", "neutral");

    try {
      const data = await apiRequest("/register", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      saveSession(data.user);
      setStatus(authStatus, "Account created. Entering workspace.", "success");
      window.location.href = "/chat";
    } catch (error) {
      setStatus(authStatus, error.message, "error");
    } finally {
      setButtonBusy(registerSubmit, false, "Creating...");
    }
  });
}
