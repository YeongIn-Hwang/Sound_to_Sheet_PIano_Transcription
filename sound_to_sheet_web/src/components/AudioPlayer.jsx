function AudioPlayer({ audioSrc }) {
  return (
    <div className="audio-player">
      <audio controls>
        <source src={audioSrc} type="audio/mpeg" />
        Your browser does not support the audio element.
      </audio>
    </div>
  );
}