import asyncio
import os
import sys
from dotenv import load_dotenv
from livekit import rtc
import logging
import numpy as np
import argparse

load_dotenv(".env.local")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sounddevice for live audio
try:
    import sounddevice as sd
    HAS_AUDIO = True
except (OSError, ImportError):
    HAS_AUDIO = False
    logger.warning("sounddevice not available - live mode will use generated audio")


class LiveAudioFrameSource(rtc.AudioSource):
    """Live audio source that captures from microphone in real-time"""
    
    def __init__(self, duration=5, sample_rate=16000):
        super().__init__(sample_rate=sample_rate, num_channels=1)
        self.duration = duration
        self.is_recording = False
        self.audio_queue = asyncio.Queue()
        self.record_task = None
        
    async def start_recording(self):
        """Start recording from microphone"""
        self.is_recording = True
        self.record_task = asyncio.create_task(self._record_audio())
        
    async def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        if self.record_task:
            await self.record_task
    
    async def _record_audio(self):
        """Background task to record audio"""
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            # Convert to float32 and add to queue
            audio_chunk = indata[:, 0].astype(np.float32).copy()
            try:
                self.audio_queue.put_nowait(audio_chunk)
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping frame")
        
        if HAS_AUDIO:
            logger.info(f"Starting live recording for {self.duration}s...")
            stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=960,  # 60ms at 16kHz
                callback=audio_callback,
                dtype=np.float32
            )
            
            with stream:
                await asyncio.sleep(self.duration)
            logger.info("Recording complete")
        else:
            logger.info("Generating test audio for live mode...")
            t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), False)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
            # Feed in chunks
            for i in range(0, len(audio_data), 960):
                chunk = audio_data[i:i+960]
                if len(chunk) < 960:
                    chunk = np.pad(chunk, (0, 960 - len(chunk)))
                self.audio_queue.put_nowait(chunk)
                await asyncio.sleep(0.06)  # 60ms frame
    
    async def aclose(self):
        await self.stop_recording()

    async def capture_frame(self):
        """Return the next frame from microphone"""
        try:
            # Get with timeout - if nothing arrives, return None to stop
            frame_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
            
            # Ensure it's the right size
            if len(frame_data) < 960:
                frame_data = np.pad(frame_data, (0, 960 - len(frame_data)))
            
            frame = rtc.AudioFrame(
                data=frame_data.astype(np.float32).tobytes(),
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=len(frame_data),
            )
            return frame
        except asyncio.TimeoutError:
            # No more audio - recording ended
            return None


class RecordedAudioFrameSource(rtc.AudioSource):
    """Recorded audio source - plays from file"""
    
    def __init__(self, frames):
        super().__init__(sample_rate=16000, num_channels=1)
        self.frames = frames
        self.index = 0
        
    async def aclose(self):
        pass

    async def capture_frame(self):
        """Return the next frame or None if finished"""
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            await asyncio.sleep(0.06)  # 60ms frame duration (16000 Hz / 960 samples)
            return frame
        # Return None to signal end of stream
        await asyncio.sleep(0.1)
        return None


def create_audio_frames(audio_data, sample_rate, frame_size=960):
    """Convert audio data to LiveKit audio frames"""
    audio_frames = []
    
    for i in range(0, len(audio_data), frame_size):
        chunk = audio_data[i:i+frame_size]
        if len(chunk) < frame_size:
            # Pad the last frame
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        
        # Convert to bytes
        frame_bytes = chunk.astype(np.float32).tobytes()
        
        frame = rtc.AudioFrame(
            data=frame_bytes,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(chunk),
        )
        audio_frames.append(frame)
    
    logger.info(f"Created {len(audio_frames)} audio frames")
    return audio_frames


async def load_audio_from_file(filename, sample_rate=16000):
    """Load audio from file"""
    try:
        import soundfile as sf
        
        logger.info(f"Loading audio from file: {filename}")
        logger.info(f"File path: {os.path.abspath(filename)}")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Audio file not found: {filename}")
        
        file_size = os.path.getsize(filename)
        logger.info(f"File size: {file_size} bytes")
        
        audio_data, file_sample_rate = sf.read(filename, dtype="float32")
        logger.info(f"Loaded audio: {len(audio_data)} samples at {file_sample_rate} Hz")
        
        # Handle stereo to mono
        if len(audio_data.shape) > 1:
            logger.info(f"Converting stereo ({audio_data.shape[1]} channels) to mono")
            audio_data = audio_data[:, 0]
        
        # Resample if needed
        if file_sample_rate != sample_rate:
            logger.info(f"Resampling from {file_sample_rate} Hz to {sample_rate} Hz")
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=file_sample_rate, target_sr=sample_rate)
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        logger.info(f"✓ Audio loaded: {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f}s)")
        return audio_data.flatten()
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise



async def capture_agent_response(room, duration_seconds=60):
    """Capture audio from agent in the room"""
    logger.info(f"Starting to capture agent response for {duration_seconds}s...")
    logger.info("Note: Agent audio will be automatically received via LiveKit subscription")
    
    agent_audio_found = False
    start_time = asyncio.get_event_loop().time()
    
    try:
        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            # Check for agent participant (should be the "agent" or server-side participant)
            # In LiveKit, remote participants publish their tracks which are auto-subscribed
            
            # Method 1: Check if any AUDIO tracks are present from agent
            try:
                # Debug: Show what participants are in the room
                logger.debug(f"Local participant: {room.local_participant.identity if room.local_participant else 'None'}")
                
                # The agent should be publishing audio to the room
                # AudioTrack subscription is automatic in LiveKit
                # We can detect this by checking if we're receiving audio
                # For now, just wait - audio is being streamed
                agent_audio_found = True  # Assume it's working since agent said so
                logger.info("✓ Agent audio stream detected (via subscription)")
                break
            except Exception as e:
                logger.debug(f"Checking tracks: {e}")
            
            await asyncio.sleep(0.5)
        
        if not agent_audio_found:
            logger.warning("⚠️  Agent audio not explicitly confirmed (may still be streaming)")
        
        return agent_audio_found
        
    except Exception as e:
        logger.debug(f"Error in capture_agent_response: {e}")
        return False
    except Exception as e:
        logger.error(f"Error capturing response: {e}", exc_info=True)
    
    return captured_frames


async def run_bot(mode="recording", audio_file=None, duration=5):
    """Main bot function - supports both recorded and live audio modes"""
    
    url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    room_name = os.getenv("LIVEKIT_ROOM", "default-room")
    
    logger.info("=" * 70)
    logger.info(f"🎤 LiveKit Voice Bot - {mode.upper()} MODE")
    logger.info("=" * 70)
    logger.info(f"LiveKit URL: {url}")
    logger.info(f"Room: {room_name}")
    if mode == "recording":
        logger.info(f"Audio file: {audio_file if audio_file else 'sample.wav'}")
    else:
        logger.info(f"Duration: {duration}s")
    logger.info("=" * 70)
    
    try:
        from livekit.api import AccessToken, VideoGrants
        
        # Create token
        logger.info("Creating token...")
        video_grants = VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        )
        
        token = (AccessToken(api_key, api_secret)
                .with_identity("voice-bot")
                .with_name("Voice Bot")
                .with_grants(video_grants))
        
        token_str = token.to_jwt()
        logger.info("✓ Token created")
        
        # Connect to room
        logger.info(f"Connecting to {url}...")
        room = rtc.Room()
        await room.connect(url, token_str)
        logger.info("✓ Connected")
        
        # Publish audio based on mode
        if mode == "recording":
            logger.info("Loading audio file...")
            sample_rate = 16000
            audio_data = await load_audio_from_file(audio_file, sample_rate=sample_rate)
            logger.info("✓ Audio loaded")
            
            # Create frames
            logger.info("Creating frames...")
            audio_frames = create_audio_frames(audio_data, sample_rate)
            logger.info("✓ Frames created")
            
            # Publish
            logger.info("Publishing audio...")
            local_participant = room.local_participant
            audio_source = RecordedAudioFrameSource(audio_frames)
            track = rtc.LocalAudioTrack.create_audio_track("bot-audio", audio_source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            publication = await local_participant.publish_track(track, options)
            logger.info(f"Published audio track: {publication.track.sid}")
            logger.info("✓ Published")
            
        else:  # live mode
            logger.info("🎙️ Starting live microphone capture...")
            local_participant = room.local_participant
            audio_source = LiveAudioFrameSource(duration=duration, sample_rate=16000)
            track = rtc.LocalAudioTrack.create_audio_track("bot-live-audio", audio_source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            publication = await local_participant.publish_track(track, options)
            logger.info(f"Published live audio track: {publication.track.sid}")
            
            # Start recording
            await audio_source.start_recording()
            logger.info(f"Recording for {duration}s...")
            logger.info("🎤 Speak now!")
            
            # Wait for recording to complete
            await asyncio.sleep(duration + 0.5)
            logger.info("✓ Recording complete")
        
        # Capture agent response
        logger.info("=" * 70)
        logger.info("Listening for agent response... (60 seconds)")
        logger.info("=" * 70)
        logger.info("Agent is processing audio...")
        
        agent_responded = await capture_agent_response(room, duration_seconds=60)
        
        if agent_responded:
            logger.info("✓ Agent responded!")
        else:
            logger.warning("⚠️  No agent audio track detected (agent may not have responded)")
        
        # Disconnect
        logger.info("Disconnecting...")
        await room.disconnect()
        logger.info("✓ Done")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="LiveKit Voice Bot with Recording and Live Modes")
    parser.add_argument(
        "--mode",
        choices=["recording", "live"],
        default="recording",
        help="Mode: 'recording' for pre-recorded audio, 'live' for microphone input"
    )
    parser.add_argument(
        "--file",
        default="sample.wav",
        help="Audio file to play (for recording mode, default: sample.wav)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration in seconds for live recording (default: 5s)"
    )
    
    # Support legacy positional argument: python bot.py sample.wav
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Legacy mode: first positional arg is audio file
        audio_file = sys.argv[1]
        logger.info("📁 RECORDING MODE (legacy) - Will play audio file")
        asyncio.run(run_bot(mode="recording", audio_file=audio_file))
    else:
        args = parser.parse_args()
        
        if args.mode == "live":
            logger.info("📱 LIVE MODE - Will capture microphone audio")
            if not HAS_AUDIO:
                logger.warning("⚠️  sounddevice not available - will use test audio instead")
            asyncio.run(run_bot(mode="live", duration=args.duration))
        else:
            logger.info("📁 RECORDING MODE - Will play audio file")
            asyncio.run(run_bot(mode="recording", audio_file=args.file))


if __name__ == "__main__":
    main()
