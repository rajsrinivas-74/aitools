#!/usr/bin/env python3
"""
Simple test to debug bot-agent communication flow
"""
import asyncio
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
import logging

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_room_state():
    """Connect to room and check state"""
    url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    room_name = os.getenv("LIVEKIT_ROOM", "default-room")
    
    logger.info(f"Connecting to {url}...")
    
    # Create access token
    token = (AccessToken(api_key, api_secret)
            .with_identity("test-bot")
            .with_name("Test Bot")
            .with_grants(VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )))
    
    # Connect to room
    room = rtc.Room()
    
    async def on_participant_connected(p: rtc.RemoteParticipant):
        logger.info(f"🟢 Participant connected: {p.identity} ({p.name})")
        for track_id, track_info in p.tracks.items():
            logger.info(f"   Track: {track_id} - {track_info.kind.name}")
    
    async def on_participant_disconnected(p: rtc.RemoteParticipant):
        logger.info(f"🔴 Participant disconnected: {p.identity}")
    
    async def on_track_subscribed(track: rtc.RemoteAudioTrack, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logger.info(f"📩 Audio track subscribed from {participant.identity}: {track.sid}")
    
    room.on_participant_connected += on_participant_connected
    room.on_participant_disconnected += on_participant_disconnected
    room.on_track_subscribed += on_track_subscribed
    
    await room.connect(url, token.to_jwt())
    logger.info(f"✓ Connected to room: {room_name}")
    logger.info(f"  Local participant: {room.local_participant.identity}")
    
    # Check who else is in the room
    logger.info(f"\nCurrent participants in room ({len(room.remote_participants.__dict__)}):")
    for pid, p in room.remote_participants:
        logger.info(f"  - {p.identity} ({p.name}): {len(p.tracks)} tracks")
    
    # Wait and observe
    logger.info("\nWaiting 10 seconds to observe room activity...")
    await asyncio.sleep(10)
    
    logger.info(f"\nFinal state:")
    logger.info(f"  Total remote participants: {len(list(room.remote_participants))}")
    for pid, p in room.remote_participants:
        logger.info(f"  - {p.identity}: {len(p.tracks)} tracks")
    
    await room.disconnect()
    logger.info("✓ Disconnected")

if __name__ == "__main__":
    asyncio.run(test_room_state())
