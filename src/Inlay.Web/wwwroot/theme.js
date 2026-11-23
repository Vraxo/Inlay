(() => {
    const storageKey = "inlay-theme";
    const systemPreference = window.matchMedia("(prefers-color-scheme: dark)");

    function getPreference() {
        return localStorage.getItem(storageKey) || "system";
    }

    function applyTheme(preference) {
        const dark = preference === "dark" ||
            (preference === "system" && systemPreference.matches);

        document.documentElement.dataset.theme = dark ? "dark" : "light";
        document.documentElement.style.colorScheme = dark ? "dark" : "light";
        return dark;
    }

    function toggleTheme() {
        const dark = document.documentElement.dataset.theme !== "dark";
        localStorage.setItem(storageKey, dark ? "dark" : "light");
        return applyTheme(dark ? "dark" : "light");
    }

    systemPreference.addEventListener("change", () => {
        if (getPreference() === "system")
            applyTheme("system");
    });

    applyTheme(getPreference());

    window.inlay = {
        getThemePreference: getPreference,
        isDarkMode: () => document.documentElement.dataset.theme === "dark",
        setThemePreference: preference => {
            const normalized = ["light", "dark"].includes(preference) ? preference : "system";
            localStorage.setItem(storageKey, normalized);
            return applyTheme(normalized);
        },
        toggleTheme
    };
})();
