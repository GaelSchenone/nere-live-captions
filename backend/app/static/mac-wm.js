// Mini window manager estilo Mac OS clasico: ventanas arrastrables, redimensionables,
// que se pueden cerrar/reabrir (menu Ventanas) y maximizar (zoom box).
// La pagina en si nunca scrollea -- cada ventana scrollea su .mac-body si hace falta.
const MacWM = (function () {
  const windows = new Map();
  let zTop = 10;
  let windowsMenuEl = null;

  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  function storageKey(id) { return `macwm:${location.pathname}:${id}`; }

  function loadRect(id) {
    try {
      const raw = localStorage.getItem(storageKey(id));
      if (raw) return JSON.parse(raw);
    } catch (e) {}
    return null;
  }

  function saveRect(win) {
    const el = win.el;
    try {
      localStorage.setItem(storageKey(win.id), JSON.stringify({
        left: parseFloat(el.style.left) || 0,
        top: parseFloat(el.style.top) || 0,
        width: el.style.width ? parseFloat(el.style.width) : null,
        height: el.style.height ? parseFloat(el.style.height) : null,
        hidden: el.hidden,
      }));
    } catch (e) {}
  }

  function bringToFront(win) {
    zTop += 1;
    win.el.style.zIndex = zTop;
    windows.forEach((w) => w.el.classList.toggle('focused', w === win));
  }

  function menuBarHeight() {
    return document.querySelector('.mac-menubar')?.offsetHeight || 22;
  }

  function makeDraggable(win) {
    const titleBar = win.titleBarEl;
    let dragging = false, startX = 0, startY = 0, startLeft = 0, startTop = 0;

    titleBar.addEventListener('mousedown', (e) => {
      if (e.target.closest('.mac-close, .mac-zoom')) return;
      dragging = true;
      bringToFront(win);
      startX = e.clientX;
      startY = e.clientY;
      startLeft = parseFloat(win.el.style.left) || 0;
      startTop = parseFloat(win.el.style.top) || 0;
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      let newLeft = startLeft + (e.clientX - startX);
      let newTop = startTop + (e.clientY - startY);
      newTop = clamp(newTop, 0, window.innerHeight - menuBarHeight() - 24);
      newLeft = clamp(newLeft, -win.el.offsetWidth + 80, window.innerWidth - 80);
      win.el.style.left = newLeft + 'px';
      win.el.style.top = newTop + 'px';
    });

    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      saveRect(win);
    });
  }

  function makeResizable(win) {
    const handle = document.createElement('div');
    handle.className = 'mac-resize-handle';
    win.el.appendChild(handle);
    let resizing = false, startX = 0, startY = 0, startW = 0, startH = 0;

    handle.addEventListener('mousedown', (e) => {
      resizing = true;
      bringToFront(win);
      startX = e.clientX;
      startY = e.clientY;
      const rect = win.el.getBoundingClientRect();
      startW = rect.width;
      startH = rect.height;
      e.preventDefault();
      e.stopPropagation();
    });

    window.addEventListener('mousemove', (e) => {
      if (!resizing) return;
      const newW = Math.max(win.minW || 240, startW + (e.clientX - startX));
      const newH = Math.max(win.minH || 120, startH + (e.clientY - startY));
      win.el.style.width = newW + 'px';
      win.el.style.height = newH + 'px';
    });

    window.addEventListener('mouseup', () => {
      if (!resizing) return;
      resizing = false;
      saveRect(win);
    });
  }

  function toggleZoom(win) {
    const el = win.el;
    if (el.dataset.zoomed === '1') {
      const r = win._preZoom;
      if (r) {
        el.style.left = r.left + 'px';
        el.style.top = r.top + 'px';
        el.style.width = r.width + 'px';
        el.style.height = r.height + 'px';
      }
      el.dataset.zoomed = '0';
    } else {
      const rect = el.getBoundingClientRect();
      win._preZoom = { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
      el.style.left = '10px';
      el.style.top = '10px';
      el.style.width = (window.innerWidth - 20) + 'px';
      el.style.height = (window.innerHeight - menuBarHeight() - 20) + 'px';
      el.dataset.zoomed = '1';
    }
    bringToFront(win);
    saveRect(win);
  }

  function renderWindowsMenu() {
    if (!windowsMenuEl) return;
    windowsMenuEl.innerHTML = '';
    windows.forEach((win) => {
      const item = document.createElement('div');
      item.className = 'mac-menu-item';
      item.textContent = (win.el.hidden ? '    ' : '✓ ') + win.title;
      item.addEventListener('click', () => {
        if (win.el.hidden) {
          showWindow(win.id);
        } else {
          win.el.hidden = true;
          saveRect(win);
          renderWindowsMenu();
        }
      });
      windowsMenuEl.appendChild(item);
    });
  }

  function showWindow(id) {
    const win = windows.get(id);
    if (!win) return;
    win.el.hidden = false;
    bringToFront(win);
    saveRect(win);
    renderWindowsMenu();
  }

  function registerWindow(el, opts) {
    const id = opts.id;
    const titleBarEl = el.querySelector('.mac-titlebar');
    const closeEl = el.querySelector('.mac-close');
    const zoomEl = el.querySelector('.mac-zoom');
    const win = {
      id,
      el,
      titleBarEl,
      title: opts.title || id,
      minW: opts.minW,
      minH: opts.minH,
    };
    windows.set(id, win);

    const saved = loadRect(id);
    el.style.left = (saved ? saved.left : opts.x) + 'px';
    el.style.top = (saved ? saved.top : opts.y) + 'px';
    el.style.width = ((saved && saved.width) || opts.w) + 'px';
    el.style.height = ((saved && saved.height) || opts.h) + 'px';
    el.hidden = saved ? !!saved.hidden : !!opts.startHidden;

    el.addEventListener('mousedown', () => bringToFront(win));
    makeDraggable(win);
    makeResizable(win);

    if (closeEl) {
      closeEl.addEventListener('click', (e) => {
        e.stopPropagation();
        el.hidden = true;
        saveRect(win);
        renderWindowsMenu();
      });
    }
    if (zoomEl) {
      zoomEl.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleZoom(win);
      });
    }

    bringToFront(win);
    renderWindowsMenu();
    return win;
  }

  function setWindowsMenuEl(el) {
    windowsMenuEl = el;
    renderWindowsMenu();
  }

  function initMenuBar(root) {
    root = root || document;
    const menus = root.querySelectorAll('.mac-menu');
    menus.forEach((menu) => {
      const label = menu.querySelector('.mac-menu-label');
      label.addEventListener('click', (e) => {
        e.stopPropagation();
        const wasOpen = menu.classList.contains('open');
        menus.forEach((m) => m.classList.remove('open'));
        if (!wasOpen) menu.classList.add('open');
      });
    });
    document.addEventListener('click', () => menus.forEach((m) => m.classList.remove('open')));

    const appleLogo = root.querySelector('.apple-logo');
    if (appleLogo) {
      appleLogo.style.cursor = 'default';
      appleLogo.addEventListener('click', () => showWindow('about'));
    }

    const clockEl = root.querySelector('.mac-clock');
    if (clockEl) startClock(clockEl);
  }

  function startClock(el) {
    function tick() {
      const now = new Date();
      const h = now.getHours() % 12 || 12;
      const m = String(now.getMinutes()).padStart(2, '0');
      const ampm = now.getHours() >= 12 ? 'PM' : 'AM';
      el.textContent = `${h}:${m} ${ampm}`;
    }
    tick();
    setInterval(tick, 1000 * 15);
  }

  function initFromDOM() {
    initMenuBar(document);
    const menuEl = document.getElementById('windows-menu');
    if (menuEl) setWindowsMenuEl(menuEl);
    document.querySelectorAll('.mac-window[data-win-id]').forEach((el) => {
      registerWindow(el, {
        id: el.dataset.winId,
        title: el.dataset.winTitle || el.dataset.winId,
        x: parseFloat(el.dataset.winX) || 40,
        y: parseFloat(el.dataset.winY) || 40,
        w: parseFloat(el.dataset.winW) || 320,
        h: parseFloat(el.dataset.winH) || 240,
        minW: parseFloat(el.dataset.winMinW) || 240,
        minH: parseFloat(el.dataset.winMinH) || 140,
        startHidden: el.hasAttribute('data-win-start-hidden'),
      });
    });
  }

  return {
    registerWindow, showWindow, setWindowsMenuEl, renderWindowsMenu,
    initMenuBar, bringToFront, initFromDOM,
  };
})();
