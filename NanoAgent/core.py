import json
import openai
import tiktoken
from .logger import DebugLogger

class NanoAgent:
    def __init__(self,api_key:str,base_url:str,model:str,max_tokens:int,actions=[],debug=False):                
        self.actions = ['think_more','end_answer'] + [action.__name__ for action in actions if callable(action)]
        self.actions_instructions = [action.__doc__ for action in actions if callable(action)]
        self.llm=openai.Client(api_key=api_key, base_url=base_url)
        self.model=model
        self.sysprmt=f'you are a logical assistant and you solve the user request with planning and execution step by step,MUST end every anwser with an action from {self.actions}.'
        self.msg=[{"role": "system", "content": self.sysprmt}]
        self.max_tokens=max_tokens
        self.debug = debug
        self.logger = DebugLogger(debug)
        self.end_msg={"role": "user", "content": "output the final answer"}

    def act_builder(self,query):
        sysprmt = f'''Actions Intro:
{'\n- '.join([f'- {action}' for action in self.actions_instructions])}
- think_more: input the user's request for more thinking.
- end_answer: output the final answer,no input needed.

Your task:
From these actions {self.actions}, convert the user's action choice into json format like:
{{
    "reason": "reason for choosing the action",
    "action": "actionName",
    "input": "actionInput"
}}'''
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sysprmt},
                {"role": "user", "content": query}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
        
    def act_executor(self,actionName:str,actionInput:str):
        if actionName=='think_more':
            return 'Analyze: '+actionInput
        else:
            return eval(actionName+'('+actionInput+')')

    def run(self,query):
        self.msg.append({"role": "user", "content": query})
        while self.msg[-2]!=self.end_msg:
            answer=''
            response=self.llm.chat.completions.create(
                model=self.model,
                messages=self.msg,
                stream=True
            )
            
            self.logger.log('user', query)
            self.logger.log('assistant', '', end='')
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    answer += content
                    print(content, end='', flush=True)
            
            self.msg.append({"role": "assistant", "content": answer})
            act = self.act_builder(answer)
            act = json.loads(act)
            
            self.logger.print('\n')
            self.logger.log('action', f"{act['action']}({act['input']})")
            self.logger.log('reason', act['reason'])
            self.logger.print('\n')
            
            # Use cl100k_base encoding for non-OpenAI models
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            if act['action']=='end_answer' or len(encoding.encode(self.msg[-1]['content'])) >= self.max_tokens:
                self.msg.append(self.end_msg)
            else:
                next_prompt = self.act_executor(act['action'], act['input'])
                self.logger.log('next_prompt', next_prompt)
                self.logger.print('\n')
                self.msg.append({"role": "user", "content": next_prompt})