import math
from pprint import pprint
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode, Resize 
import xformers
TOKENSCON = 77
TOKENS = 75
import torch.nn as nn

##Modified from RPG

from torch.nn.functional import interpolate
def adjust_latent_to_x(user_latent, target_shape, target_device, target_dtype):
    """
    Adjust user_latent to match the shape and type of target (x).
    
    Args:
        user_latent (torch.Tensor): Input latent tensor.
        target_shape (tuple): Shape of the target tensor.
        target_device (torch.device): Device of the target tensor.
        target_dtype (torch.dtype): Data type of the target tensor.

    Returns:
        torch.Tensor: Adjusted latent tensor.
    """
    batch_size, target_tokens, target_embedding_dim = target_shape

    # 현재 user_latent의 크기
    curr_batch, curr_tokens, curr_embedding_dim = user_latent.shape

    # 1. Spatial 토큰 크기 맞춤
    if curr_tokens != target_tokens:
        user_latent = torch.nn.functional.interpolate(
            user_latent.permute(0, 2, 1),  # (batch, embedding_dim, tokens)
            size=target_tokens,  # 목표 토큰 크기
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)  # (batch, tokens, embedding_dim)

    # 2. Embedding 크기 맞춤
    if curr_embedding_dim != target_embedding_dim:
        embedding_resize = nn.Linear(curr_embedding_dim, target_embedding_dim).to(
            target_device, dtype=target_dtype
        )
        user_latent = embedding_resize(user_latent)

    # 3. 데이터 타입 맞춤
    if user_latent.dtype != target_dtype:
        user_latent = user_latent.to(target_dtype)

    return user_latent



def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value,attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states


def main_forward_diffusers(module,hidden_states,encoder_hidden_states,divide,userpp = False,tokens=[],width = 64,height = 64,step = 0, isxl = False, inhr = None):
    context = encoder_hidden_states
    query = module.to_q(hidden_states)
    # cond, uncond =query.chunk(2)
    # query=torch.cat([cond,uncond])
    key = module.to_k(context)
    value = module.to_v(context)
    query = module.head_to_batch_dim(query)
    key = module.head_to_batch_dim(key)
    value = module.head_to_batch_dim(value)
    hidden_states=_memory_efficient_attention_xformers(module, query, key, value)
    hidden_states = hidden_states.to(query.dtype)
    # linear proj
    hidden_states = module.to_out[0](hidden_states)
    # dropout
    hidden_states = module.to_out[1](hidden_states)
    return hidden_states
    
    
    
    
    
def hook_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "Attention":
            # print(f"Attaching hook to {name}")
            module.forward = hook_forward(self, module)           

def adjust_latent_to_region(latent, start, end, latent_h, latent_w):
    """Region에 맞게 latent 크기 조정"""
    h_start = int(latent_h * start)
    h_end = int(latent_h * end)
    return latent[:, h_start:h_end, :]

def hook_forward(self, module):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = hidden_states
        print("x shape", x.shape)
        context = encoder_hidden_states
        user_latents = getattr(self, "user_latents", None)
        height = self.h
        width = self.w
        x_t = x.size()[1]
        scale = round(math.sqrt(height * width / x_t))
        latent_h = round(height / scale)
        latent_w = round(width / scale)
        ha, wa = x_t % latent_h, x_t % latent_w
        if ha == 0:
            latent_w = int(x_t / latent_h)
        elif wa == 0:
            latent_h = int(x_t / latent_w)
        
        # 사용자 latent 적용
        #if user_latents is not None:
            #print("[DEBUG] Applying user latents...")
            #x = self.apply_user_latents(x, user_latents, latent_h, latent_w)  # 올바르게 호출

        contexts = context.clone()
        def matsepcalc(x, contexts, user_latents, divide):
            print("x shape", x.shape)
            h_states = []
            x_t = x.size()[1]
            latent_h, latent_w = split_dims(x_t, height, width, self)

            latent_out = latent_w
            latent_in = latent_h

            tll = self.pt
            i = 0
            outb = None
            region_idx = 0

            # Base context 처리
            if self.usebase:
                context = contexts[:, tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON, :]
                cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                if cnet_ext > 0:
                    context = torch.cat([context, contexts[:, -cnet_ext:, :]], dim=1)
                i += 1
                print("x before main forward diffusers", x.shape)
                outb = main_forward_diffusers(module, x, context, divide, userpp=True, isxl=self.isxl)
                print("out before clone", outb.shape)
                outb = outb.clone()
                outb = outb.reshape(outb.size()[0], latent_h, latent_w, outb.size()[2])
                print(f"[DEBUG] Base context processed: outb.shape={outb.shape}")
            else:
                outb = None
                outb_t = None

            sumout = 0

            # Region-specific latent 처리
            region_idx = 0  # 추가: user_latents 인덱싱용
            for drow in self.split_ratio:
                v_states = []
                sumin = 0
                for dcell in drow.cols:
                    print(f"[DEBUG] user_latents type: {type(user_latents)}")
                    print(f"[DEBUG] user_latents length: {len(user_latents)}")

                    # 사용자 제공 latent가 있으면 해당 값 사용
                    if user_latents and region_idx < len(user_latents):
                        print(f"[DEBUG] Using user_latent for region_idx={region_idx}")
                        out = user_latents[region_idx]
                        print("user latent out before adjustment", out.shape)

                        # x와 shape이 맞지 않을 경우 조정
                        if out.shape[1:] != x.shape[1:]:
                            print(f"[DEBUG] Adjusting user_latent from shape {out.shape} to match x shape {x.shape}")
                            out = adjust_latent_to_x(out, x.shape, x.device, x.dtype)
                            print(f"[DEBUG] Adjusted user_latent shape: {out.shape}")

                        region_idx += 1
                    else:
                        # 기존 방식으로 context 처리
                        context = contexts[:, tll[i][0] * TOKENSCON:tll[i][1] * TOKENSCON, :]
                        cnet_ext = contexts.shape[1] - (contexts.shape[1] // TOKENSCON) * TOKENSCON
                        if cnet_ext > 0:
                            context = torch.cat([context, contexts[:, -cnet_ext:, :]], dim=1)
                        i += 1 + dcell.breaks

                        out = main_forward_diffusers(module, x, context, divide, userpp=self.pn, isxl=self.isxl)
                        print("out shape before resize", out.shape)

                    # reshape
                    out = out.reshape(out.size()[0], latent_h, latent_w, out.size()[2])

                    addout = 0
                    addin = 0
                    sumin += int(latent_in * dcell.end) - int(latent_in * dcell.start)
                    if dcell.end >= 0.999:
                        addin = sumin - latent_in
                        sumout += int(latent_out * drow.end) - int(latent_out * drow.start)
                        if drow.end >= 0.999:
                            addout = sumout - latent_out

                    out = out[
                        :,
                        int(latent_h * drow.start) + addout:int(latent_h * drow.end),
                        int(latent_w * dcell.start) + addin:int(latent_w * dcell.end),
                        :
                    ]

                    if self.usebase and outb is not None:
                        outb_t = outb[
                            :,
                            int(latent_h * drow.start) + addout:int(latent_h * drow.end),
                            int(latent_w * dcell.start) + addin:int(latent_w * dcell.end),
                            :
                        ].clone()
                        out = out * (1 - dcell.base) + outb_t * dcell.base

                    v_states.append(out)

                output_x = torch.cat(v_states, dim=2)
                h_states.append(output_x)

            output_x = torch.cat(h_states, dim=1)
            output_x = output_x.reshape(x.size()[0], x.size()[1], x.size()[2])
            print(f"[DEBUG] Final output shape: {output_x.shape}")

            return output_x


        # Forward 계산
        if x.size()[0] == 1 * self.batch_size:
            output_x = matsepcalc(x, contexts, user_latents, divide=1)
        else:
            if self.isvanilla:
                nx, px = x.chunk(2)
                conn, conp = contexts.chunk(2)
            else:
                px, nx = x.chunk(2)
                conp, conn = contexts.chunk(2)

            opx = matsepcalc(px, conp, user_latents, divide=2)
            onx = matsepcalc(nx, conn, user_latents, divide=2)

            if self.isvanilla:
                output_x = torch.cat([onx, opx])
            else:
                output_x = torch.cat([opx, onx])

        self.pn = not self.pn
        self.count = 0
        return output_x

    return forward



def split_dims(x_t, height, width, self=None):
    """Split an attention layer dimension to height + width.
    The original estimate was latent_h = sqrt(hw_ratio*x_t),
    rounding to the nearest value. However, this proved inaccurate.
    The actual operation seems to be as follows:
    - Divide h,w by 8, rounding DOWN.
    - For every new layer (of 4), divide both by 2 and round UP (then back up).
    - Multiply h*w to yield x_t.
    There is no inverse function to this set of operations,
    so instead we mimic them without the multiplication part using the original h+w.
    It's worth noting that no known checkpoints follow a different system of layering,
    but it's theoretically possible. Please report if encountered.
    """
    scale = math.ceil(math.log2(math.sqrt(height * width / x_t)))
    latent_h = repeat_div(height, scale)
    latent_w = repeat_div(width, scale)
    if x_t > latent_h * latent_w and hasattr(self, "nei_multi"):
        latent_h, latent_w = self.nei_multi[1], self.nei_multi[0] 
        while latent_h * latent_w != x_t:
            latent_h, latent_w = latent_h // 2, latent_w // 2

    return latent_h, latent_w

def repeat_div(x,y):
    """Imitates dimension halving common in convolution operations.
    
    This is a pretty big assumption of the model,
    but then if some model doesn't work like that it will be easy to spot.
    """
    while y > 0:
        x = math.ceil(x / 2)
        y = y - 1
    return x