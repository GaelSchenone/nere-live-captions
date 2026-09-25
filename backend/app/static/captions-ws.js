// WebSocket a /ws/captions/{session} con reconexion automatica.
// Si el server se reinicia o se corta la red, reintenta solo cada 2s en vez
// de dejar la pantalla muerta hasta que alguien la refresque a mano --
// esto va a estar prendido durante una charla en vivo, no puede fallar en silencio.
function connectCaptions(sessionId, onMessage, onStatusChange) {
  let ws = null;
  let reconnectTimer = null;
  let stopped = false;

  function open() {
    if (stopped || !sessionId) return;
    const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${wsProto}://${location.host}/ws/captions/${sessionId}`);

    ws.onopen = () => { if (onStatusChange) onStatusChange('connected'); };

    ws.onmessage = (evt) => {
      try { onMessage(JSON.parse(evt.data)); } catch (e) { /* mensaje invalido, ignorar */ }
    };

    ws.onclose = () => {
      if (onStatusChange) onStatusChange('disconnected');
      if (!stopped) reconnectTimer = setTimeout(open, 2000);
    };

    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }

  open();

  return {
    stop() {
      stopped = true;
      clearTimeout(reconnectTimer);
      if (ws) ws.close();
    },
  };
}
