class FakeUsage:
    def __init__(self):
        self.prompt_tokens     = 142
        self.completion_tokens = 87

class FakeOpenAIResponse:
    def __init__(self):
        self.usage = FakeUsage()
        self.model = "gpt-4o"


from token_ledger import configure, track_llm_cost

configure(
    project_name = "test_run",
    budget       = 1.0,
    output_dir   ="..",
    print_each   = True,
    save_on_exit = True,
)

@track_llm_cost(model="gpt-4o", name="fake_call")
def call_fake_llm():
    return FakeOpenAIResponse()

from token_ledger import get_session

if __name__ == "__main__":

    n = 5
    for i in range(0,n):
        call_fake_llm()

    session = get_session()
    print("After ",n, " calls:")
    print("Total cost  :", session.total_cost)
    print("Total tokens:", session.total_tokens)