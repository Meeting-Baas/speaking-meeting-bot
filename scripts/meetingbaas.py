import argparse
import asyncio
import os
import os
from datetime import datetime

import aiohttp
import pytz
from dotenv import load_dotenv
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.vad import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.aggregators.llm_response import OpenAILLMContext
from pipecat.serializers import ProtobufFrameSerializer
from pipecat.services.tts import CartesiaTTSService
from pipecat.services.stt import DeepgramSTTService

# from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.services.llm import OpenAILLMService
from pipecat.transports.websocket import (
    WebsocketClientParams,
    WebsocketClientTransport,
)
from pipecat.audio.base import BaseAudioResampler

from config.persona_utils import PersonaManager
from config.prompts import DEFAULT_SYSTEM_PROMPT
from meetingbaas_pipecat.utils.logger import configure_logger

load_dotenv(override=True)

logger = configure_logger()


# Function tool implementations
async def get_weather(
    function_name, tool_call_id, arguments, llm, context, result_callback
):
    """Get the current weather for a location."""
    location = arguments["location"]
    format = arguments["format"]  # Default to Celsius if not specified
    unit = (
        "m" if format == "celsius" else "u"
    )  # "m" for metric, "u" for imperial in wttr.in

    url = f"https://wttr.in/{location}?format=%t+%C&{unit}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                weather_data = await response.text()
                await result_callback(
                    f"The weather in {location} is currently {weather_data} ({format.capitalize()})."
                )
            else:
                await result_callback(
                    f"Failed to fetch the weather data for {location}."
                )


async def get_time(
    function_name, tool_call_id, arguments, llm, context, result_callback
):
    """Get the current time for a location."""
    location = arguments["location"]

    # Set timezone based on the provided location
    try:
        timezone = pytz.timezone(location)
        current_time = datetime.now(timezone)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        await result_callback(f"The current time in {location} is {formatted_time}.")
    except pytz.UnknownTimeZoneError:
        await result_callback(
            f"Invalid location specified. Could not determine time for {location}."
        )


async def main(
    meeting_url: str = "",
    persona_name: str = "Meeting Bot",
    entry_message: str = "Hello, I am the meeting bot",
    bot_image: str = "",
    streaming_audio_frequency: str = "24khz",
    websocket_url: str = "",
    enable_tools: bool = True,
):
    """
    Run the MeetingBaas bot with specified configurations

    Args:
        meeting_url: URL to join the meeting
        persona_name: Name to display for the bot
        entry_message: Message to send when joining
        bot_image: URL for bot avatar
        streaming_audio_frequency: Audio frequency for streaming (16khz or 24khz)
        websocket_url: Full WebSocket URL to connect to, including any path
        enable_tools: Whether to enable function tools like weather and time
    """
    # Load environment variables for credentials (OpenAI, etc.)
    load_dotenv()

    # Validate WebSocket URL
    if not websocket_url:
        logger.error("Error: WebSocket URL not provided")
        return

    logger.info(f"Using WebSocket URL: {websocket_url}")

    # Extract bot_id from the websocket_url if possible
    # Format is usually: ws://localhost:{PORT}/pipecat/{client_id} or the ngrok URL
    parts = websocket_url.split("/")
    # Dynamically determine the expected port for localhost URLs
    expected_local_port = os.getenv("PORT", "7014")
    if "localhost" in websocket_url and f":{expected_local_port}/pipecat/" in websocket_url:
        bot_id = parts[-1] if len(parts) > 3 else "unknown"
    elif "ngrok.io" in websocket_url:
        # Assume ngrok URL will have the client_id as the last part after /pipecat/
        bot_id = parts[-1] if len(parts) > 3 and parts[-2] == "pipecat" else "unknown"
    else:
        # Fallback for other URL formats or if client_id is not easily extractable
        bot_id = parts[-1] if len(parts) > 3 else "unknown"
    logger.info(f"Using bot ID: {bot_id}")

    # Set sample rate based on streaming_audio_frequency
    output_sample_rate = 24000 if streaming_audio_frequency == "24khz" else 16000
    # Silero VAD only supports 16000 or 8000 Hz
    vad_sample_rate = 16000

    logger.info(
        f"Using audio frequency: {streaming_audio_frequency} (output sample rate: {output_sample_rate}, VAD sample rate: {vad_sample_rate})"
    )

    # Create resampler for VAD if needed
    resampler = None
    if output_sample_rate != vad_sample_rate:
        resampler = BaseAudioResampler()
        logger.info(f"Created resampler for converting {output_sample_rate}Hz to {vad_sample_rate}Hz")

    # Set up the WebSocket transport with correct sample rates - use the full WebSocket URL directly
    transport = WebsocketClientTransport(
        uri=websocket_url,
        params=WebsocketClientParams(
            audio_out_sample_rate=output_sample_rate,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16000,  # Must be either 8000 or 16000
                params=VADParams(
                    threshold=0.5,  # Confidence threshold for speech detection
                    min_speech_duration_ms=250,  # Time in ms that speech must be detected
                    min_silence_duration_ms=100,  # Time in ms of silence required
                    min_volume=0.6,  # Minimum audio volume threshold
                ),
            ),
            vad_audio_passthrough=True,
            serializer=ProtobufFrameSerializer(),
            timeout=300, 
        ),
    )

    # Get persona configuration
    persona_manager = PersonaManager()
    persona = persona_manager.get_persona(persona_name)
    if not persona:
        logger.error(f"Persona '{persona_name}' not found")
        return

    # Get additional content from persona
    additional_content = persona.get("additional_content", "")
    if additional_content:
        logger.info("Found additional content for persona")
    else:
        logger.info("No additional content found for persona")

    # Get voice ID from persona if available, otherwise use env var
    voice_id = persona.get("cartesia_voice_id") or os.getenv("CARTESIA_VOICE_ID")
    logger.info(f"Using voice ID: {voice_id}")

    # Initialize services
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=voice_id,  # Use voice ID from persona
        sample_rate=output_sample_rate,  # Use the same sample rate as transport
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview",
        run_in_parallel=False,
    )

    # Register function tools if enabled
    if enable_tools:
        logger.info("Registering function tools")

        # Register functions
        llm.register_function("get_weather", get_weather)
        llm.register_function("get_time", get_time)

        # Define function schemas
        weather_function = FunctionSchema(
            name="get_weather",
            description="Get the current weather",
            properties={
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            required=["location", "format"],
        )

        time_function = FunctionSchema(
            name="get_time",
            description="Get the current time for a specific location",
            properties={
                "location": {
                    "type": "string",
                    "description": "The location for which to retrieve the current time (e.g., 'Asia/Kolkata', 'America/New_York')",
                },
            },
            required=["location"],
        )

        # Create tools schema
        tools = ToolsSchema(standard_tools=[weather_function, time_function])
    else:
        logger.info("Function tools are disabled")
        tools = None

    # Add speech-to-text service
    # Extract language code from persona if available
    language = persona.get("language_code", "en-US")
    logger.info(f"Using language: {language}")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        encoding="linear16" if streaming_audio_frequency == "16khz" else "linear24",
        sample_rate=output_sample_rate,
        language=language,  # Use language from persona
    )
    # stt = GladiaSTTService(
    #     api_key=os.getenv("GLADIA_API_KEY"),
    #     encoding="linear16" if streaming_audio_frequency == "16khz" else "linear24",
    #     sample_rate=output_sample_rate,
    #     language=language,  # Use language from persona
    # )

    # Make sure we're setting a valid bot name
    bot_name = persona_name or "Bot"
    logger.info(f"Using bot name: {bot_name}")

    # Create a more comprehensive system prompt
    system_content = persona["prompt"]

    # Add additional context if available
    if additional_content:
        system_content += f"\n\nYou are {persona_name}\n\n{DEFAULT_SYSTEM_PROMPT}\n\n"
        system_content += "You have the following additional context. USE IT TO INFORM YOUR RESPONSES:\n\n"
        system_content += additional_content
        system_content += "You are a meeting bot. You are in a meeting with a group of people. You are here to help the group. You are not the host of the meeting. You are not the organizer of the meeting. You are not the participant in the meeting. You are the meeting bot."
        system_content += "YOU ARE HELP TO HELP. KEEP IT SHORT. EVERYTHING YOU SAY WILL BE REPEATED BACK TO THE GROUP OUT LOUD so DO NOT add PUNCTUATION OR CAPS. JUST SAY WHAT YOU NEED TO SAY IN A CONCISE MANNER."


    # Set up messages
    messages = [
        {
            "role": "system",
            "content": system_content,
        },
    ]

    # Create the context object - with or without tools
    if enable_tools and tools:
        context = OpenAILLMContext(messages, tools)
    else:
        context = OpenAILLMContext(messages)

    # Get the context aggregator pair using the LLM's method
    # This handles properly setting up the context aggregators
    aggregator_pair = llm.create_context_aggregator(context)

    # Get the user and assistant aggregators from the pair
    user_aggregator = aggregator_pair.user()
    assistant_aggregator = aggregator_pair.assistant()

    # Create pipeline using the single instances - adding STT service
    pipeline = Pipeline(
        [
            transport.input(),
            stt,  # Add speech-to-text service
            user_aggregator,  # Process user input and update context
            llm,
            tts,
            transport.output(),
            assistant_aggregator,  # Store LLM responses in context
        ]
    )

    # Create and run task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True, check_dangling_tasks=True))
    runner = PipelineRunner()

    # Handle the initial greeting if needed
    if entry_message:
        logger.info("Bot will speak first with an introduction")
        # Prepare a greeting message
        initial_message = {
            "role": "user",
            "content": entry_message,
        }

        # Queue the initial message to be processed once the pipeline starts
        async def queue_initial_message():
            await asyncio.sleep(2)  # Small delay to ensure transport is ready
            await task.queue_frames([LLMMessagesFrame([initial_message])])
            logger.info("Initial greeting message queued")

        # Create a task to queue the initial message
        asyncio.create_task(queue_initial_message())

    # Run the pipeline
    await runner.run(task)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a MeetingBaas bot")
    parser.add_argument("--meeting-url", help="URL of the meeting to join")
    parser.add_argument(
        "--persona-name", default="Meeting Bot", help="Name to display for the bot"
    )
    parser.add_argument(
        "--entry-message",
        default="Hello, I am the meeting bot",
        help="Message to send when joining",
    )
    parser.add_argument("--bot-image", default="", help="URL for bot avatar")
    parser.add_argument(
        "--streaming-audio-frequency",
        default="16khz",
        choices=["16khz", "24khz"],
        help="Audio frequency for streaming (16khz or 24khz)",
    )
    parser.add_argument(
        "--websocket-url", help="Full WebSocket URL to connect to, including any path"
    )
    parser.add_argument(
        "--enable-tools",
        action="store_true",
        help="Enable function tools like weather and time",
    )

    args = parser.parse_args()

    # Run the bot
    asyncio.run(
        main(
            meeting_url=args.meeting_url,
            persona_name=args.persona_name,
            entry_message=args.entry_message,
            bot_image=args.bot_image,
            streaming_audio_frequency=args.streaming_audio_frequency,
            websocket_url=args.websocket_url,
            enable_tools=args.enable_tools,
        )
    )
