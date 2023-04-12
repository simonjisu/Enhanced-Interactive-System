# import uvicorn
# if __name__ == '__main__':
#     uvicorn.run('app.main:app', host='localhost', port=8080, reload=True)

import os
import json
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler

from pathlib import Path 
from dotenv import load_dotenv
from flask import Flask, request
from managers import DatabaseManager, ChatManager, PromptManager, KnowledgeManager
from models import *
from collections import defaultdict



# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

DB_MANAGER = DatabaseManager(db_path=os.environ['DB_PATH'])
CHAT_MANAGER = ChatManager(openai_api_key=os.environ['OPENAI_API_KEY'], chat_model=os.environ['CHAT_MODEL'])
PROMPT_MANAGER = PromptManager()
KNOWLEDGE_MANAGER = KnowledgeManager(knowledge_path=os.environ['KNOWLEDGE_PATH'])
SESSION_STATUS = defaultdict()

# SLACK
app = App(
    token=os.environ['SLACK_TOKEN'],
    signing_secret=os.environ['SIGNING_SECRET']
)

@app.event('app_mention')
def event_test(event, client, say):
    user_id = event['user']
    bot_id = client.api_call("auth.test")['user_id']
    raw_text = event['text']
    if raw_text.startswith(f'<@{bot_id}>'):
        text = raw_text.lstrip(f'<@{bot_id}>')
            
        if SESSION_STATUS.get(user_id) is None:
            # must start a new session
            responses = [{'chat_response': {'content': f'Please start a new session by typing `/session new`'}}]
        else:
            responses = PROMPT_MANAGER.main(
                user_id=user_id,
                query=text, 
                chat_manager=CHAT_MANAGER, 
                db_manager=DB_MANAGER, 
                knowledge_manager=KNOWLEDGE_MANAGER
            )
        
        for response in responses:
            chat_response = response['chat_response']
            # say: send a message to the channel
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"```{chat_response['content']}```" if chat_response['type'] == 'dataframe' else f"{chat_response['content']}"
                    },
                }
            ]
            if response.get('code'):
                code_response = response['code']
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Code:\n```{code_response['content']}```" if code_response['type'] == 'codeblock' else f"{code_response['content']}"
                    }
                })
            if response.get('data'):
                data_response = response['data']
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Input Data:\n```{data_response['content']}```" if data_response['type'] == 'dataframe' else f"{data_response['content']}"
                    }
                })
            say(blocks=blocks, text='Contents')

@app.command('/help')
def repeat_text(ack, respond):
    # Acknowledge command 
    ack()
    respond(f"""
Type `/session new` to start a new session
Type `/session end` to end the current session
Type `/session status` to check the status of the current session
Type `/session restart` to restart the current session
Type `/help` to see this message again

After start the session, please call `@Assistant` to interaction with system.
""")

@app.command('/session')
def new_session(ack, respond, command):
    ack()
    user_id = command['user_id']
    if command.get('text') == 'new':
        CHAT_MANAGER.new_session(user_id=user_id)
        if user_id not in SESSION_STATUS:
            SESSION_STATUS[user_id] = defaultdict()
            SESSION_STATUS[user_id]['status'] = True
        # respond: send a message to only to the user
        respond(f'New session started for <@{user_id}>:\nHow Can I Help You?')
    elif command.get('text') == 'status':
        respond(json.dumps(SESSION_STATUS, indent=4))
        # if user_id in SESSION_STATUS:
        #     respond(f'The session is up')
        # else:
        #     respond(f'No current session for <@{user_id}>')
    elif command.get('text') == 'restart':
        if user_id in SESSION_STATUS:
            SESSION_STATUS.pop(user_id)
            SESSION_STATUS[user_id] = defaultdict()
            SESSION_STATUS[user_id]['status'] = True
            respond(f'Restart session for <@{user_id}>')
        else:
            respond(f'No current session for <@{user_id}>')
    elif command.get('text') == 'end':
        if user_id in SESSION_STATUS:
            SESSION_STATUS[user_id] = defaultdict()
        respond(f'End session for <@{user_id}>')
    else:
        respond('Invalid command: type `/session new` or `/session end`')

# FLASK
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.route('/help', methods=['POST'])
def help_handler():
    return handler.handle(request)

@flask_app.route('/session', methods=['POST'])
def session_handler():
    return handler.handle(request)

if __name__ == '__main__':
    flask_app.run(debug=True, port=5005)
    # SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()