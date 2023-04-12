import openai
from datetime import datetime as dt
from collections import defaultdict
import sqlite3
import pandas as pd
import json
from typing import Any
from pprint import pprint
import re
import yaml
import time
import joblib
from collections import defaultdict


NUM_ERROR_TRIALS = 3
TIME_FMT = '%Y-%m-%d %H:%M:%S'

def extract_code_block(text):
    code_block_pattern = re.compile(r"```(?:\w*\n)?(.*?)\n?```", re.DOTALL)
    code_block = code_block_pattern.search(text)
    if code_block:
        return f"```\n{code_block.group(1)}\n```"
    else:
        return ''


class DatabaseManager():
    def __init__(self, db_path='database.sqlite'):
        self.db_path = db_path
        self.dbtype = 'sqlite' 

    def start(self):
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        self.conn.close()

    def get_table_description(self, to_str=False):
        self.start()
        cur = self.conn.cursor()
        res = cur.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()
        tables = list(map(lambda x: x[0], res))
        # select column names
        db_info = {}
        for t in tables:
            db_info[t] = []
            res = cur.execute(f'PRAGMA table_xinfo({t})').fetchall()
            for r in res:
                db_info[t].append(
                    {'column': r[1], 'type': r[2], 'pk': bool(r[5])}
                )
        cur.close()
        self.close()

        if to_str:
            s = ''
            for i, (k, v) in enumerate(db_info.items()):
                s += f'[Table: {k}] '
                for j, col in enumerate(v):
                    s += f'{col["column"]} {col["type"]}'
                    if col['pk']:
                        s += ' (pk)'
                    if j != len(v) - 1:
                        s += ', '
                if i != len(db_info) - 1:
                    s += '\n'
 
            return s
        else:
            return db_info
    
    def summarize_table(self, table_name):
        """Should be defined by the creator, e.g. number of the row, accessable dates, etc. 
        Show this to user to know what do we have in the table."""
        cur = self.conn.cursor()
        cur.close()
        return 

    def exec_query(self, query, rt_pandas=True):
        self.start()
        try:
            cur = self.conn.cursor()
            res = cur.execute(query)
            if rt_pandas:
                columns = [x[0] for x in res.description]
                df = pd.DataFrame(res.fetchall(), columns=columns)
                cur.close()
                self.close()
                return df
            else:
                cur.close()
                self.close()
                return res.fetchall()
        except Exception as e:
            self.close()
            return e
        

class KnowledgeManager():
    def __init__(self, knowledge_path='knowledges.yml'):
        self.knowledge_path = knowledge_path
        with open(self.knowledge_path, 'r') as file:
            self.knowledges = yaml.safe_load(file)

    def get_custom_knowledge(self, filter_key=None, to_str: bool=False):
        if filter_key is not None:
            ks = {k: dict(filter(lambda x: x[0] == filter_key, v.items())) for k, v in self.knowledges.items()}
        else:
            ks = self.knowledges

        if to_str:
            return json.dumps(ks)
        else:
            return ks
        
    def get_knowledge_keys(self):
        ks_dict = {'[0]': 'Doesn not exists similar keyword'}
        ks = list(self.knowledges.keys())
        for i, x in enumerate(ks, 1):
            ks_dict[f'[{i}]'] = f'{x}'
        # return dict(map(lambda x: (f'[{x[0]}]', f'{x[1]}'), enumerate(ks)))
        return ks_dict

class RoleContentDict(dict):
    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)
        self.role = role  # for openai api
        self.content = content  # for openai api


class ChatManager():
    def __init__(self, openai_api_key: str, chat_model='gpt-3.5-turbo'):
        openai.api_key = openai_api_key
        self.llm = openai.ChatCompletion
        self.chat_model = chat_model
        self.usage_counter = defaultdict(int)
        self.messages: dict[str, list[RoleContentDict]] = defaultdict()

    def new_session(self, user_id: str=None):
        self.messages[user_id] = [
            RoleContentDict(**{"role": "system", "content": "Hi! I'm a chatbot. Ask me anything about the database."})
        ]
        # self.messages = [{"role": "system", "content": "Hi! I'm a chatbot. Ask me anything about the database."}]

    def update_context(self, user_id: str, role_content_dict: RoleContentDict):
        # keep tracking the messages
        self.messages[user_id].append(role_content_dict)

    def update_usage_counter(self, usage_dict: RoleContentDict):
        for k, v in usage_dict.items():
            self.usage_counter[k] += v

    def query(self, user_id: str, query: str) -> RoleContentDict:
        self.update_context(
            user_id=user_id,
            role_content_dict=RoleContentDict(**{"role": "user", "content": query})
        )
        msg = self.llm.create(
            model=self.chat_model,
            messages=self.messages[user_id],
        )
        res_role_content = RoleContentDict(**msg['choices'][0]['message'])
        self.update_context(
            user_id=user_id,
            role_content_dict=res_role_content
        )
        self.update_usage_counter(usage_dict=msg['usage'])
        time.sleep(1)
        return res_role_content

class PromptManager():
    message_types = {
        '[1]': {'name': 'information_retrieval', 'code_type': 'SQL'}, 
        '[2]': {'name': 'prediction', 'code_type': 'python, SQL'},
        '[3]': {'name': 'others', 'code_type': 'none'}
    }

    def new_session(self, user_id: str, chat_manager: ChatManager):
        chat_manager.new_session(user_id=user_id)
    
    def feed_table_description(self, user_id: str, chat_manager: ChatManager, db_manager: DatabaseManager):
        table_desc = db_manager.get_table_description(to_str=True)
        prompt_contents = f'Based on {db_manager.dbtype} Database Table and Column Description:\n {table_desc}'
        prompt_role = 'user'
        chat_manager.update_context(
            user_id=user_id,
            role_content_dict=RoleContentDict(**{'role': prompt_role, 'content': prompt_contents})
        )

    def feed_custom_knowledge(self, user_id: str, key: str, chat_manager: ChatManager, knowledge_manager: KnowledgeManager):
        # now we only have formula as knowledge to serve
        ks = knowledge_manager.knowledges[key]['formula']
        promt_contents = f'Please refer Knowledges:\n {ks}'
        prompt_role = 'user'
        chat_manager.update_context(
            user_id=user_id,
            role_content_dict=RoleContentDict(**{'role': prompt_role, 'content': promt_contents})
        )

    def extract_keyword_knowledge(self, user_id: str, query: str, chat_manager: ChatManager, knowledge_manager: KnowledgeManager):
        """Extract keyword from user's query."""
        keys = knowledge_manager.get_knowledge_keys()  # dict  [0] [1] ...
        keys_items_str = ' '.join([f'{k} {v}' for k, v in keys.items()])
        keys_str = ','.join(keys.keys())
        query_str = """
        Extract the a target word(what user want to know) from the user's question: {query}

        Select from following options: {keys_items_str}
        Please answer only with format {keys_str}. Don't need to explain.
        """.format(query=query, keys_items_str=keys_items_str, keys_str=keys_str)
        role_content = chat_manager.query(
            user_id=user_id,
            query=query_str
        )
        answer = '[' + re.findall(r'\d', role_content.content)[0] + ']'
        print('extracted answer', answer)
        print(keys)
        key_to_use = keys.get(answer)  # If answer is not in keys, return None
        return key_to_use
    
    def intention_classification(self, user_id: str, query: str, chat_manager: ChatManager):
        """
        Given user's query classifies the intention of the query.

        Query can be classified with: 
        - Information Retrieval: sql
        - Prediction: embedded ml
        
        1. public / private
        2. non-time related / time related
        3. Information Retrieval / Prediction 
        """

        query_str = '''Today is {cur_time}\nUser's Question: {query}
What is user's intention?
Here are the options:
    1. Information Retrival(SQL)?
    2. Prediction(ML)?
    3. Others(Including all are not related to the database and prediction question, e.g., greetings, thanks, etc.)
Please answer only with format [1], [2] or [3]. Don't need to explain.'''.format(
            cur_time=dt.now().strftime('%Y-%m-%d'), query=query)
        role_content = chat_manager.query(
            user_id=user_id,
            query=query_str
        )
        return role_content

    def generate_code(self, user_id: str, intention_message: str, query: str, chat_manager: ChatManager, knowledge=None) -> RoleContentDict:
        if intention_message == '[1]':
            code_type = self.message_types[intention_message]['code_type']
            query_str = '''Please generate the code(language: {code_type}) for the user's question: {query}
Write the code in the following code block format.
Do not write any text explanation. Only write the executable the code in text format.
```[placehoder for code]```'''.format(code_type=code_type, query=query)
        elif intention_message == '[2]':
            # Load Data
            query_str = '''Here is the preprocess code:
```
{load_data}
```

Please generate the data loading code(language: SQL) for the user's question: {userquery}
Write SELECT clause and keep the same column name in the preprocess code.

Write the code in the following code block format.
Do not write any text explanation. Only write the executable code in text format.
```[placehoder for code]```
'''.format(load_data=knowledge['LoadData'], userquery=query)
        else:
            return chat_manager.query(user_id, query)
        
        role_content = chat_manager.query(user_id, query_str)
        msg = role_content['content']
        print('------------ Prompt --------------')
        print(query_str)
        print('------------ Result --------------')
        print(msg)
        msg = extract_code_block(msg)
        codes = re.sub(r'```(?:[a-zA-Z]+\n)?|\b```', '', msg) 
        code = codes.split(';')[0]   # Force to run only first block of code
        return RoleContentDict(role=role_content['role'], content=code)

    def execute_code(self, intention_message: str, code: str, db_manager: DatabaseManager, **kwragrs):
        if intention_message == '[1]':
            res = db_manager.exec_query(code, rt_pandas=True)
            return res, None
        elif intention_message == '[2]':
            data = db_manager.exec_query(code, rt_pandas=True)
            model = kwragrs['model']
            res = model.predict(data)
            return pd.DataFrame(res), data
        elif intention_message == '[3]':
            return None, None
        else:
            raise ValueError('Invalid intention message')

    def handle_execution_error(self, user_id: str, error: Exception, chat_manager: ChatManager):
        error_msg = f'[{type(error).__module__}.{type(error).__name__}] {error}'
        query_str = f'''Execution Error: {error_msg} Please check the code. 
Do not write any text explanation. Only write the executable the code in text format.
```[placehoder for code]```'''
        role_content = chat_manager.query(user_id, query_str)
        msg = role_content['content']
        code = extract_code_block(msg)
        code = re.sub(r'```(?:[a-zA-Z]+\n)?|\b```', '', code)  # .lstrip('```').rstrip('```')
        print(code)
        return RoleContentDict(role=role_content['role'], content=code)

    def execute_error_correction(
            self, user_id: str, code: str, intention_message: str, key: str|None,
            chat_manager: ChatManager, db_manager: DatabaseManager, 
            knowledge_manager: KnowledgeManager):
        i = 0
        handled = False
        while i < NUM_ERROR_TRIALS:
            print(f'----- [Handle Execution Error] Trials: {i+1}-----')
            print(executed_results)
            self.feed_table_description(user_id, chat_manager, db_manager)
            if key is not None:
                self.feed_custom_knowledge(user_id, key, chat_manager, knowledge_manager)
            executed_results = self.handle_execution_error(
                user_id=user_id, 
                error=executed_results, 
                chat_manager=chat_manager
            )
            code = executed_results.content
            executed_results, data = self.execute_code(
                intention_message=intention_message,
                code=code,
                db_manager=db_manager
            )
            time.sleep(1)
            if not isinstance(executed_results, Exception):
                handled = True
                break
            i += 1

        return handled, code, executed_results, data

    def main(self, user_id: str, query: str, chat_manager: ChatManager, db_manager: DatabaseManager, knowledge_manager: KnowledgeManager) -> RoleContentDict:
        print('----- Intention Classiciation -----')
        intention_role_content = self.intention_classification(user_id, query, chat_manager)
        check_intention = re.findall(r'\d', intention_role_content.content)  # list
        if len(check_intention) > 0:
            intention_message = '[' + check_intention[0] + ']'
        else:
            intention_message = '[3]'
        
        print(f'INTENTION MESSAGE = {intention_message}')
        pprint(intention_role_content)
        all_outputs = []
        if intention_message == '[1]':
            self.feed_table_description(user_id, chat_manager, db_manager)
            print('----- Extract Keywords -----')
            key = self.extract_keyword_knowledge(user_id, query, chat_manager, knowledge_manager)
            if key is not None:
                self.feed_custom_knowledge(user_id, key, chat_manager, knowledge_manager)
                print(key)
            print('----- Generate Code -----')
            answer = self.generate_code(
                user_id=user_id,
                intention_message=intention_message, 
                query=query, 
                chat_manager=chat_manager
            )
            code = answer.content
            
            if '' == code:
                role_content = chat_manager.query(user_id, query)
                all_outputs.append({
                    'chat_response': {
                        'type': 'text',
                        'content': role_content['content']},
                })
            else:
                executed_results, _ = self.execute_code(
                    intention_message=intention_message, 
                    code=code, 
                    db_manager=db_manager
                )
                # Handle Error
                if isinstance(executed_results, Exception):
                    handled, code, executed_results, _ = self.execute_error_correction(
                        user_id=user_id,
                        code=code,
                        intention_message=intention_message,
                        key=key,
                        chat_manager=chat_manager,
                        db_manager=db_manager,
                        knowledge_manager=knowledge_manager
                    )
                    
                    if not handled:
                        all_outputs.append({
                            'chat_response': {
                                'type': 'str',
                                'content': 'Sorry, I cannot handle this error. Please check the code.'
                            },
                            'code': {
                                'type': 'codeblock',
                                'content': code
                            }
                        })
                                            
                else:    
                    # Output
                    if isinstance(executed_results, pd.DataFrame):
                        out_results = executed_results.to_markdown(index=False)
                        out_type = 'dataframe'
                    else:
                        out_results = str(executed_results)
                        out_type = 'str'
                    
                    all_outputs.append({
                        'chat_response': {
                            'type': out_type,
                            'content': out_results
                        },
                        'code': {
                            'type': 'codeblock',
                            'content': code
                        }
                    })
        elif intention_message == '[2]':
            print('----- Extract Keywords -----')
            key = self.extract_keyword_knowledge(user_id, query, chat_manager, knowledge_manager)
            if (key is None) or (not knowledge_manager.knowledges.get(key)) or \
                    ('model_path' not in list(knowledge_manager.knowledges.get(key).keys()) and \
                    'knowledge_path' not in list(knowledge_manager.knowledges.get(key).keys())):
                all_outputs.append({
                    'chat_response': {
                        'type': 'str',
                        'content': 'Sorry, There is no model avaliable.'
                    },
                })
            else:
                ks = knowledge_manager.knowledges[key]  # contains formula, model_path, knowledge_path 

                # model_path, knowledge_path
                print(ks)
                model = joblib.load(ks['model_path']['forest'])
                knowledge = joblib.load(ks['knowledge_path'])
                print('----- Generate Code: LoadData -----')
                answer = self.generate_code(user_id, intention_message, query, chat_manager, knowledge=knowledge)
                code = answer.content
                executed_results, data = self.execute_code(intention_message, code, db_manager, model=model)
                # Handle Error
                if isinstance(executed_results, Exception):
                    handled, code, executed_results, data = self.execute_error_correction(
                        user_id=user_id,
                        code=code,
                        intention_message=intention_message,
                        key=key,
                        chat_manager=chat_manager,
                        db_manager=db_manager,
                        knowledge_manager=knowledge_manager
                    )
                    
                    if not handled:
                        all_outputs.append({
                            'chat_response': {
                                'type': 'str',
                                'content': 'Sorry, I cannot handle this error. Please check the code.'
                            },
                            'code': {
                                'type': 'codeblock',
                                'content': code
                            }
                        })   
                else:    
                    # Output
                    if isinstance(executed_results, pd.DataFrame):
                        out_results = executed_results.to_markdown(index=False)
                        out_data = data.to_markdown(index=False)
                        out_type = 'dataframe'
                    else:
                        out_results = str(executed_results)
                        out_data = str(data)
                        out_type = 'str'

                    all_outputs.append({
                        'chat_response': {
                            'type': out_type,
                            'content': out_results},
                        'data': {
                            'type': out_type,
                            'content': out_data},
                        })
        else:
            role_content = chat_manager.query(user_id, query)
            return [{
                'chat_response': {
                    'type': 'text',
                    'content': role_content['content']},
            }]
        
        print('----- All Outputs -----')
        pprint(all_outputs)
        return all_outputs  # list[dict]