import json,time
from .logger import DebugLogger
from datetime import datetime
import openai
import tiktoken

class NanoAgent:
    def __init__(self,api_key:str,base_url:str,model:str,max_tokens:int,actions=[],debug=False,retry=20):                
        self.actions = ['think_more','final_result'] + [action.__name__ for action in actions if callable(action)]
        self.actions_instructions = [action.__doc__ for action in actions if callable(action)]
        self.llm=openai.Client(api_key=api_key, base_url=base_url)
        self.model=model
        self.sysprmt=f'you are a logical assistant, you can destructively analyze the user request including the deep purpose and the reasoning, then output step by step execution, MUST end every anwser with an action from {self.actions} until this answer is the final result.'
        self.msg=[{"role": "system", "content": self.sysprmt}]
        self.max_tokens=max_tokens
        self.max_retries=retry
        self.debug = debug
        self.logger = DebugLogger(debug)
        self.language = None
        self.end_msg={"role": "user", "content": "output the final result with proper format"}
        self.save_path = None
        self.action_format = {
            "action": "actionName",
            "input": "actionInput",
            "lang": "language of the user query"
        }

    def act_builder(self,query:str)->dict:
        sysprmt = f'''Actions Intro:
{'\n- '.join([f'- {action}' for action in self.actions_instructions])}
- think_more: take a deep breath and think more about the user query.
- final_result: action is final_result, input is "".

Your task:
From these actions {self.actions}, based on the user query, output the user's next action in json format :
{str(self.action_format)}'''
        self.logger.log('sysprmt', sysprmt)
        retry=self.max_retries
        while retry>0:
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                    {"role": "system", "content": sysprmt},
                    {"role": "user", "content": query}
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
                    if not all(k in result for k in self.action_format.keys()):
                        self.logger.log('error', f"Invalid action received, will retry\n{result}\n")
                        continue
                if self.language is None and result['lang'] is not None:
                    self.language = result['lang']
                    self.end_msg["content"] = "output the final result in language "+self.language
                return result
            except Exception as e:
                retry-=1
                self.logger.log('error', e)
                time.sleep(30)
                continue
        
    def act_executor(self,actionName:str,actionInput:str):
        if actionName=='think_more':
            return f'Take a deep breath and think more about: {actionInput} in language {self.language}'
        else:
            return eval(actionName+'('+actionInput+')')

    def save_msg(self, filename=None):
        """Save conversation messages to a JSON file"""
        if not filename:
            # Generate filename using datetime and first 14 chars of last user query
            last_query = next((msg['content'] for msg in reversed(self.msg) if msg['role'] == 'user'), '')[:14]
            timestamp = datetime.now().strftime('%d%m%Y_%H%M')
            filename = f"{timestamp}_{last_query}.json"
        
        self.save_path = filename
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.msg, f, ensure_ascii=False, indent=2)

    def run(self, query):
        if query.endswith('.json'):
            with open(query, 'r', encoding='utf-8') as f:
                self.msg = json.load(f)
            self.save_path = query
        else:
            self.msg.append({"role": "user", "content": query})
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
            # Save after assistant message
            self.save_msg(self.save_path)
            
            act = self.act_builder(answer)
            
            self.logger.log('action', f"\n{act['action']}({act['input']})")
            
            # Use cl100k_base encoding for non-OpenAI models
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            if act['action']=='final_result' or len(encoding.encode(self.msg[-1]['content'])) >= self.max_tokens:
                self.msg.append(self.end_msg)
            else:
                next_prompt = self.act_executor(act['action'], act['input'])
                self.logger.log('next_prompt', f"\n{next_prompt}\n")
                self.msg.append({"role": "user", "content": next_prompt})
            self.save_msg(self.save_path)