"""GLM Coding Plan through the user-selected Anthropic Messages endpoint.
Uses application-scoped credentials; never reads or edits global Claude settings.
"""
import base64
import json
import re
import httpx
from app.model_providers.glm_api import GlmApiAdapter
from app.model_providers.base import ReviewResponse

BASE_URL = 'https://open.bigmodel.cn/api/anthropic'
MODEL = 'glm-5.3-flash'

def eligibility_status():
    # Legacy DB field only; successful capability testing governs activation.
    return 'confirmed'

class GlmCodingAdapter(GlmApiAdapter):
    account_kind='coding_plan'
    protocol='anthropic'

    def __init__(self,transport=None,base_url=BASE_URL):
        super().__init__(transport=transport,base_url=base_url)

    def _chat(self,api_key,model,messages,max_tokens,timeout_s,json_mode=False):
        converted=[]
        for message in messages:
            blocks=message['content']
            if isinstance(blocks,str):blocks=[{'type':'text','text':blocks}]
            content=[]
            for block in blocks:
                if block['type']=='text':content.append(block)
                elif block['type']=='image_url':
                    url=block['image_url']['url']
                    try:
                        header,data=url.split(',',1)
                        media=header[5:].split(';')[0]
                        if not header.startswith('data:') or ';base64' not in header or media not in ('image/png','image/jpeg','image/webp'):raise ValueError()
                        base64.b64decode(data,validate=True)
                    except (ValueError,TypeError):
                        return ReviewResponse(ok=False,error_code='invalid_response',error_message='图片格式无效')
                    content.append({'type':'image','source':{'type':'base64','media_type':media,'data':data}})
            converted.append({'role':message['role'],'content':content})
        body={'model':MODEL,'max_tokens':max(1024,max_tokens),'messages':converted}
        if json_mode:
            body['system']='Return only valid JSON matching the requested schema.'
            body['thinking']={'type':'disabled'}
        try:
            with self._client() as client:
                response=client.post(self._base+'/v1/messages',json=body,
                    headers={'Authorization':'Bearer '+api_key,'anthropic-version':'2023-06-01'},
                    timeout=httpx.Timeout(timeout_s,connect=10))
        except httpx.TimeoutException:
            return ReviewResponse(ok=False,error_code='timeout_unknown',error_message='请求超时，结果未知')
        except httpx.HTTPError:
            return ReviewResponse(ok=False,error_code='network_error',error_message='网络连接失败')
        code=response.status_code
        if code!=200:
            err='auth_failed' if code in (401,403) else 'rate_limited' if code==429 else 'server_error' if code>=500 else 'provider_error'
            try:
                provider_error = response.json().get('error') or {}
                business_code = str(provider_error.get('code', ''))
                if business_code == '1113':
                    err = 'quota_exhausted'
                elif business_code in ('1308', '1310'):
                    message = 'Coding Plan 使用额度已达上限，请等待额度重置后重试'
                    # Extract only a timestamp; never expose arbitrary provider text or secrets.
                    reset = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', str(provider_error.get('message', '')))
                    if reset:
                        message += f'；平台返回的重置时间：{reset.group(0)}'
                    return ReviewResponse(ok=False, error_code='plan_limit_reached', error_message=message)
                elif business_code == '1309':
                    err = 'plan_expired'
            except (ValueError,AttributeError):pass
            return ReviewResponse(ok=False,error_code=err,error_message=f'GLM 请求失败（HTTP {code}）')
        try:
            data=response.json()
            blocks=data.get('content')
            if not isinstance(blocks,list) or any(not isinstance(b,dict) for b in blocks):
                raise ValueError()
            text=''.join(b['text'] for b in blocks if b.get('type')=='text' and isinstance(b.get('text'),str))
            stop=data.get('stop_reason')
            diagnostics={'http_status':code,'stop_reason':stop if stop in ('end_turn','max_tokens','tool_use','stop_sequence','pause_turn','refusal') else 'unknown',
                         'block_types':[b.get('type') for b in blocks if b.get('type') in ('text','thinking','tool_use','redacted_thinking')],
                         'text_chars':len(text)}
            usage={k:v for k,v in (data.get('usage') or {}).items() if isinstance(v,(int,float))}
            if stop=='max_tokens':
                return ReviewResponse(ok=False,error_code='output_truncated',error_message='模型输出达到长度上限，结果不完整',usage=usage,diagnostics=diagnostics)
            if stop=='tool_use' or any(b.get('type')=='tool_use' for b in blocks):
                return ReviewResponse(ok=False,error_code='tool_required',error_message='模型要求调用视觉工具，当前直连接口未执行该工具',usage=usage,diagnostics=diagnostics)
            if not text.strip():
                return ReviewResponse(ok=False,error_code='empty_response',error_message='模型未返回可用文本',usage=usage,diagnostics=diagnostics)
            return ReviewResponse(ok=True,content=text,provider_request_id=data.get('id',''),usage=usage,diagnostics=diagnostics)
        except (ValueError,KeyError,TypeError,AttributeError):
            return ReviewResponse(ok=False,error_code='invalid_response',error_message='模型响应结构异常',diagnostics={'http_status':code})

    def probe(self,api_key):
        result=super().probe(api_key)
        result.capabilities.model_id=MODEL
        result.capabilities.structured_output=False
        result.capabilities.multi_image=False
        if result.capabilities.notes.get('synth_answer') is not None:
            result.ok=False;result.capabilities.vision_ok=False
            result.error_code='capability_missing';result.detail='文本连接可用，但视觉测试未通过'
        return result

    def review_images(self,api_key,request):
        # No hidden retries and no switch to the ordinary billed API.
        return self._chat(api_key,MODEL,[{'role':'user','content':[
            {'type':'text','text':request.prompt},*[{'type':'image_url','image_url':{'url':u}} for u in request.images]]}],
            request.max_tokens,request.timeout_s,True)
