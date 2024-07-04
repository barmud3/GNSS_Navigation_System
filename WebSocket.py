import asyncio
import websockets
import json
import struct

async def gnss_data_handler(websocket, path):
    reader, writer = await asyncio.open_connection('localhost', 5560)
    try:
        while True:
            raw_data = await reader.read(4096)  # Adjust buffer size as necessary
            if raw_data:
                data = struct.unpack('<f', raw_data)  # Example: unpacking if data is in binary format
                # Process raw_data or data as needed
                await websocket.send(json.dumps(data))  # Example: sending data to WebSocket client
    finally:
        writer.close()

async def main():
    server = await websockets.serve(gnss_data_handler, 'localhost', 8765)
    await server.wait_closed()  # Ensure server is properly closed

if __name__ == '__main__':
    asyncio.run(main())
