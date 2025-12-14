export default function StatusText({ show, text }) {
  if (!show) return null;
  return (
    <div className="no-print" style={{ marginTop: 12, opacity: 0.85 }}>
      {text}
    </div>
  );
}
