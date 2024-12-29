import json,time
from .logger import DebugLogger
from datetime import datetime
import openai
import tiktoken

class NanoAgent:
    def __init__(self,api_key:str,base_url:str,model:str,max_tokens:int,actions=[],debug=False,retry=20):                
        self.action_functions = {action.__name__: action for action in actions if callable(action)}
        self.action_instructions = [action.__doc__ for action in self.action_functions.values()]
        self.action_format = {
            "action": "actionName",
            "input": "actionInput",
            "lang": "language of the user query"
        }
        self.llm=openai.Client(api_key=api_key, base_url=base_url)
        self.model=model
        self.sysprmt=f"You are an helpful assistant that performs step by step deconstructive reasoning.\
describes the next step, you can ask user to use tools like {', '.join(self.action_functions.keys())} to help you.\
use the language of the user query\
MUST END EVERY STEP WITH ASKING THE USER TO CONFIRM THE STEP UNTIL THE USER REQUESTS THE FINAL RESULT."
        self.msg=[{"role": "system", "content": self.sysprmt}]
        self.max_tokens=max_tokens
        self.max_retries=retry
        self.debug = debug
        self.logger = DebugLogger(debug)
        self.language = None
        self.end_msg={"role": "user", "content": "output the final result with proper format"}
        self.save_path = None
        self.user_query = None

    def act_builder(self,answer:str)->dict:
        prompt = f'''<actions_intro>
{'\n- '.join([f'- {action}' for action in self.action_instructions])}
- think_more: push user to think different ways for the target,input is the suggestion.
- final_result: action is final_result, input is "".
</actions_intro>
<user_query>
{self.user_query+'\n'+answer}
</user_query>

Your task:
Based on the user query, pick next action from\
    {['think_more','final_result'] + list(self.action_functions.keys())} \
    for the user, output in json format :
    {str(self.action_format)}'''
        retry=self.max_retries
        while retry>0:
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                    {"role": "user", "content": prompt}
                ],
                    response_format={ "type": "json_object" }
                )
                result = json.loads(response.choices[0].message.content)
                if not isinstance(result, dict):
                    if isinstance(result, list) and len(result)>0 and isinstance(result[0], dict):
                        result = result[0]
                    else:
                        self.logger.log('error', f"Invalid action received, will retry\n{result}\n")
                        continue
                    if not all(k in result for k in self.action_format):
                        self.logger.log('error', f"Invalid action received, will retry\n{result}\n")
                        continue
                if self.language is None and result['lang'] is not None:
                    self.language = result['lang']
                    self.end_msg["content"] = "base on the previous steps, output the final result in language "+self.language
                return result
            except Exception as e:
                retry-=1
                self.logger.log('error', e)
                time.sleep(30)
                continue
        
    def act_exec(self,actionName:str,actionInput:str)->str:
        if actionName=='think_more':
            return f'Take a deep breath and think more about: {actionInput} \n output in language {self.language}'
        elif actionName=='final_result':
            return ''
        else:
            return self.action_functions[actionName](actionInput)

    def save_msg(self, filename=None):
        if not filename:
            first_query = self.user_query[:14]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"{timestamp}_{first_query}.json"
        
        self.save_path = filename
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.msg, f, ensure_ascii=False, indent=2)

    def run(self, query:str):
        if query.endswith('.json'):
            with open(query, 'r', encoding='utf-8') as f:
                self.msg = json.load(f)
            self.save_path = query
            self.user_query = next((msg['content'] for msg in self.msg if msg['role'] == 'user'), '')
        else:
            self.msg.append({"role": "user", "content": query})
            self.user_query = query
            self.save_msg(self.save_path)

        retry = self.max_retries
        while retry > 0:
            self.logger.print('\n')
            answer = ''
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=self.msg,
                    stream=True
                )
                self.logger.log('user', self.msg[-1]['content'])
                self.logger.log('assistant', '', end='')
                
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        answer += content
                        yield content
                self.logger.print('\n')
                
            except Exception as e:
                retry -= 1
                self.logger.log('error', e)
                time.sleep(30)
                continue

            if self.msg[-1] == self.end_msg:
                return
            
            self.msg.append({"role": "assistant", "content": answer})
            self.save_msg(self.save_path)
            
            act = self.act_builder(answer)
            self.logger.log('action', f"\n{act['action']}({act['input']})")
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            if act['action']=='final_result' or len(encoding.encode(self.msg[-1]['content'])) >= self.max_tokens:
                self.msg.append(self.end_msg)
            else:
                next_prompt = self.act_exec(act['action'], act['input'])
                self.logger.log('next_prompt', f"\n{next_prompt}\n")
                self.msg.append({"role": "user", "content": next_prompt})
            self.save_msg(self.save_path)