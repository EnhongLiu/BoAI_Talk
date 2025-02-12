
# Agent 1: Decide which tool to use
def decide_model(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to which tool to use, or simply end.
    
    Args:
        state (messages): The current state
    
    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---DECIDE WHICH TOOL TO USE---")

    message = state["messages"][-1].content
    model = llm # use llama for now, but woudl suggest to switch to GPT for better performance. <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------
    milkbot_tool = Tool(func = milkbot_prediction_visuals, name ='MilkBot_visuals', description = 'generate visuals from user query')
    tools = [milkbot_tool]
    model = model.bind_tools(tools)
    
    response = model.invoke(message)
    state["messages"].append(response)

    print('---DECISION MADE. WILL USE',response.tool_calls[0]['name'],'---')
    
    return {"messages": state['messages']}
    


# Agent 2: Process user question to extract key parameters to input the MilkBot model
class extraction(BaseModel):
    region: list[str] = Field(default_factory=lambda:['eu'])
    parity: list[str] = Field(default_factory=lambda:['1', '2', '3+'])
    dim_range: List[int] = Field(default_factory=lambda:list(np.arange(1, 306, 1)))

llm = llm
llm.with_structured_output(extraction)

def prepare_query_for_milk_yield_visuals(state):
    """
    This function prepares the query for milk yield visuals.
    Args:
        state (messages): The current state
    Returns:
        dic: A dict containing the extracted parameters
    """

    print('---EXTRACTING KEY INFORMATION FROM QUESTION---')

    query=state['messages'][0].content

    prompt = """
    Extract the following parameters from the user's query:

    1. Region (e.g., US, Europe): If the region is mentioned, extract it. If not specified, default to ['eu'].
    2. Parity (number indicating parity): Extract parity if explicitly mentioned (e.g., parity 1). If not specified, default to ['1','2','3+'].
    3. DIM range (list format): Extract if specified as a range or list. If not, default to np.arange(1, 306, 1).

    User Query: "{query}"

    Return ONLY the extracted parameters in strict JSON format without any additional text or explanations. 
    Ensure the format follows this example:
    {{"region":["US"],"parity":['1'],"dim_range":[1, 2, 3, ..., 305]}}
    """
    
    extraction_prompt = ChatPromptTemplate.from_template(prompt)
    
    chain = extraction_prompt | llm
    response = chain.invoke(query)
    state["messages"].append(response)
    
    return {"messages": state['messages']}



# Agent 3: Judge whether the extracted information meets expectations. If not, rewrite the question and loop back to Agent 1. If yes, proceed to Agent 5.
def grade_answer(state):
    """
    Determines whether the response are relevant to the question.
    Args:
        state (messages): The current state
    Returns:
        str: A decision for whether the response is relevant or not
    """
    print("---CHECKING RELEVANCE AND CORRECTNESS---")

    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = llm # This model can be modified later to be a different one according to users' preference or computation restrictioin <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
    template=
    """
    You are a grader responsible for determining if an answer correctly addresses the user's question by evaluating extracted key parameters. 

    ### Task:
    Assess if the answer aligns with the user's intent by extracting key parameters (region, parity, DIM range) from the question and verifying that the answer responds accordingly.

    ### Provided Information:
    - **Answer:** 
    {answer} 

    - **User Question:** 
    {question}

    ### Parameter Extraction Guidelines:
    Extract the following parameters from the user question:
    1. **Region** – Extract the region if explicitly mentioned (e.g., 'US', 'Europe'). If not mentioned, default to **'eu'**.
    2. **Parity** – Extract parity (e.g., 'parity 1'). If no parity is specified, default to **[1, 2, 3]**.
    3. **DIM Range** – Extract the DIM (Days in Milk) range if provided. If unspecified, default to **np.arange(1, 306, 1)**.

    ### Output Requirements:
    - Return the extracted parameters in the following **strict JSON format**: 
    ```json
    {{"region": ["Europe"], "parity": [1],"dim_range": [1, 2, 3, ..., 305]}}
    ```
    - Do not include explanations, extra text, or deviations from this format.

    ### Grading Criteria:
    - Return **"yes"** if the answer addresses the extracted parameters accurately.  
    - Return **"no"** if the answer fails to meet expectations based on extracted parameters.

    """, 
    input_variables=["answer", "question"]
)

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]

    question = messages[0].content
    answer = messages[-1].content

    scored_result = chain.invoke({"question": question, "answer": answer})

    score = scored_result.binary_score

    # print(score)

    if score == "yes":
        print("---DECISION: ANSWER MEETS EXPECTATION---")
        return "milkbot_visuals"

    else:
        print("---DECISION: ANSWER DOES NOT MEET EXPECTATION---")
        # print(score)
        return "rewrite"



# Agent 4: Rewrite the question if the extracted information from Agent 2 does not meet expectations.
def rewrite(state):
    """
    Transform the query to produce a better question.
    Args:
        state (messages): The current state
    Returns:
        dict: The updated state with re-phrased question
    """
    print("---TRANSFORM QUERY SINCE NO GOOD ANSWER WAS FOUND---")

    question = state["messages"][0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input, reason about the underlying semantic intent / meaning, and formulate ONE AND ONLY ONE improved question within 15 words. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate one and only one improved question: """,
        )
    ]
    # Rewrite agent
    model = llm  # use llama70b for now. <<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------
    response = model.invoke(msg)
    state["messages"].append(response)
    
    return {"messages": state['messages']}



# Agent 5: Generate visuals and answer from the extracted information
def milkbot_visuals(state):
    """
    Invokes the milkbot model to generate a response based on the current state. It will produce the prediction and visuals
    
    Args:
        state (messages): The current state
    
    Returns:
        dict: The updated state with the agent response appended to messages
    """

    print('---GENERATING VISUALS FROM EXTRACTED INFORMATION---')

    message = json.loads(state["messages"][-1].content)

    region_ai= message['region']
    parity_ai= message['parity']
    dim_range_ai= message['dim_range']

    # print(region_ai)
    
    my_prediction = predict_milk_yield_for_region(region=region_ai, parity=parity_ai, dim_range=dim_range_ai)

    milk_yield_plot(my_prediction)

    response = HumanMessage(content=f"--- \nHere are the graphs generated by **__Visuals@MilkBot__** for your query: \n{state['messages'][0].content} \n---")

    state['messages'].append(response)

    print(response.content)

    return {"messages": state['messages']}
