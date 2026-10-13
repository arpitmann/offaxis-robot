# Your own sounds (optional)

Drop any of these files here (wav, m4a, mp3, aac, caf, aiff or ogg), then run `python3 make_robot_app.py` again:

| File name | When the robot plays it |
|---|---|
| `welcome` | when the app opens |
| `hey_beautiful` | when it first sees you |
| `i_love_you` | when you make a heart sign or blow a kiss |
| `wake_up` | when your eyes stay shut for a couple of seconds |
| `song` | when the 🎸 button is pressed (the robot strums along to the beat) |

Anything missing is replaced by the phone's own voice, simple beeps, or (for the song) strummed chords.
Voice Memos recordings (`.m4a`) work fine. On a Mac they are converted with the built-in `afconvert`.

Files in this folder are ignored by git, so your recordings stay on your computer.
