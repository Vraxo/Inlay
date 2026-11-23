window.inlay = {
    getThemePreference: () => localStorage.getItem("inlay-theme") || "system",
    setThemePreference: preference => {
        if (preference === "system")
            localStorage.removeItem("inlay-theme");
        else
            localStorage.setItem("inlay-theme", preference);
        applyTheme(preference);
    },
    download: async (fileName, streamReference, contentType = "image/png") => {
        const stream = await streamReference.arrayBuffer();
        const blob = new Blob([stream], { type: contentType });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = fileName;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
};

function applyTheme(preference) {
    const dark = preference === "dark" ||
        (preference === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    document.querySelector('meta[name="theme-color"]')
        ?.setAttribute("content", dark ? "#000000" : "#fafafa");
}

const themePreference = localStorage.getItem("inlay-theme") || "system";
applyTheme(themePreference);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if ((localStorage.getItem("inlay-theme") || "system") === "system")
        applyTheme("system");
});
