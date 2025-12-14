export default function ErrorBox({ error }) {
  if (!error) return null;
  return (
    <div className="no-print" style={{ marginTop: 16, color: "crimson", whiteSpace: "pre-wrap" }}>
      {error}
    </div>
  );
}
