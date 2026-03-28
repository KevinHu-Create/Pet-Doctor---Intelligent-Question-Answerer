import {
  apiRequest,
  homePathForRole,
  redirectIfAuthenticated,
  saveSession,
  serializeForm,
  setButtonBusy,
  setStatus,
} from "/static/shared.js";

const alreadyAuthenticated = await redirectIfAuthenticated();

if (!alreadyAuthenticated) {
  const authStatus = document.getElementById("auth-status");
  const loginForm = document.getElementById("login-form");
  const registerForm = document.getElementById("register-form");
  const adminForm = document.getElementById("admin-form");
  const loginSubmit = document.getElementById("login-submit");
  const registerSubmit = document.getElementById("register-submit");
  const adminSubmit = document.getElementById("admin-submit");
  const tabButtons = [...document.querySelectorAll("[data-auth-tab]")];

  function setTab(mode) {
    tabButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.authTab === mode);
    });

    loginForm.classList.toggle("is-hidden", mode !== "login");
    registerForm.classList.toggle("is-hidden", mode !== "register");
    adminForm.classList.toggle("is-hidden", mode !== "admin");
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
      window.location.href = homePathForRole(data.user.role);
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
      window.location.href = homePathForRole(data.user.role);
    } catch (error) {
      setStatus(authStatus, error.message, "error");
    } finally {
      setButtonBusy(registerSubmit, false, "Creating...");
    }
  });

  adminForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = serializeForm(adminForm);

    setButtonBusy(adminSubmit, true, "Authorizing...");
    setStatus(authStatus, "Checking administrator credentials.", "neutral");

    try {
      const data = await apiRequest("/admin/login", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      saveSession(data.user);
      setStatus(authStatus, "Admin login successful. Opening console.", "success");
      window.location.href = homePathForRole(data.user.role);
    } catch (error) {
      setStatus(authStatus, error.message, "error");
    } finally {
      setButtonBusy(adminSubmit, false, "Authorizing...");
    }
  });
}
