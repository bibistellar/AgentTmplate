#react模板
import ipaddress
import langgraph
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, List, Dict, Annotated, Union
import requests
import json
from openai import OpenAI
import os
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def summarize_address_range(input_str: str) -> List[str]:
       """
       将IP地址范围划分为最少数量的CIDR块
       
       Args:
           input_str: 格式为"起始IP,结束IP"，例如'192.168.1.0,192.168.1.15'
           
       Returns:
           CIDR块列表，例如['192.168.1.0/28']
       """
       try:
           parts = input_str.split(',')
           if len(parts) != 2:
               return ["Error: 输入格式应为'起始IP,结束IP'"]
               
           start_ip, end_ip = parts
           start = ipaddress.IPv4Address(start_ip.strip())
           end = ipaddress.IPv4Address(end_ip.strip())
           networks = ipaddress.summarize_address_range(start, end)
           return [str(network) for network in networks]
       except Exception as e:
           return [f"Error: {str(e)}"]

@tool
def write_result(input_str: str) -> str:
       """
        将IP划分的方案写入到result字段中
       
       Args:
           input_str: IP划分方案
           
       Returns:
           写入成功:"Scuccess"
           写入失败:Err原因
       """
       try:
            print(input_str)
            return "Scucess"
       except Exception as e:
           return [f"Error: {str(e)}"]


# IP分配专家的角色提示
IP_EXPERT_PROMPT = """你是一位网络IP分配专家，负责为数据中心的网络设备和容器分配IP地址。

规则说明：
1. 分配的资源为公网IP，公网IP是一段连续的IP，可能是完整的CIDR，也可能不是。如果是连续的IP，应该使用summarize_address_range工具进行处理
2. 每个服务商可能会提供多个IP网段，或者多个连续的IP，请注意分辨。
2. 在多段公网IP中，应优先使用数据中心运营商的其中一个IP网段A分配给网络设备。网络设备一般包括网关、交换机、relay机、BMC等。其余的网段称为网段B。
3. 按照顺序(网关、交换机、relay机、BMC)为这些设备分配IP。
4. 对于被拿去分配IP给网络设备的IP网段A，一定要使用提供的summarize_address_range工具对网段A剩下的IP进行处理，将其划分为最优CIDR网段。在这里网段A划分出来的网段不需要指定网关。因为它们的网关实际上就是之前分配的网络设备中的网关
5. 每个网段B都需要指定一个网关IP，但这个网关IP不应该导致进一步的网段划分。例如183.245.12.64/26网段，划分为183.245.12.64/26 （网关：183.245.12.65），没有因为网关的划分导致网段的分裂
6. 将分配方案通过write_result进行输出

用户将以自然语言描述他们的IP资源和分配需求。你需要理解用户需求并生成合适的IP分配方案。按照如下的例子给出IP分配方案

例子：
台州三线集群新建
cn-taizhou5-ix
新建字节集群，16台机器（包含1台备用机） 4台交换机
移动80G：电信70G：联通50G （联通机房）
移动IPV4：183.245.12.64/26 
电信IPV4：115.231.148.0/26 
联通IPV4：103.3.113.128/26 

IP规划方案：
---------------------------------------------
联通ip：
103.3.113.128/26
103.3.113.129 网关ip
103.3.113.130 - 103.3.113.133 4台交换机管理ip
103.3.113.134 - 103.3.113.136 3台relay机ip
103.3.113.137 - 103.3.113.139 3台relay机 BMC ip

容器ip：
103.3.113.140 - 103.3.113.190 （网关 103.3.113.129）
103.3.113.140/30
103.3.113.144/28
103.3.113.160/27

---------------------------------------------
移动ip：
183.245.12.64/26 （网关：183.245.12.65）

---------------------------------------------
电信ip：
115.231.148.0/26 （网关：115.231.148.1）

如果需要将IP范围划分为最优CIDR块，你可以使用以下工具：

{tools}

记住：优先使用数据中心运营商(机房所在的运营商)的IP为网络设备分配IP地址。

用户输入: {input}

{agent_scratchpad}
"""

# 定义工具
tools = [summarize_address_range,write_result]

# 创建模型
model = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    model="qwen-plus",
    api_key="sk-2510020788c34412aa89f57be5eef11b",
    streaming=False,
)

# 使用自定义提示创建ReAct代理
graph = create_react_agent(
    model, 
    tools=tools, 
    prompt=IP_EXPERT_PROMPT
)

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# 示例运行
if __name__ == "__main__":
    # 示例输入
    inputs = {"messages": [("user", "IP资源:联通:183.95.152.0/26,183.95.152.64/27（机房IP运营商）移动：IPV4:111.47.207.0/26,111.47.207.64/27电信ip：119.96.148.0-119.96.148.159网络设备：4台交换机，3台relay机，3台relay机的BMC需要分配IP")]}
    print_stream(graph.stream(inputs, stream_mode="values"))