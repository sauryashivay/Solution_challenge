import os
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from config import paths_config
from dotenv import load_dotenv

load_dotenv()

class CustomerData(BaseModel):
    """
    Data model representing a customer's financial and personal profile.
    
    Attributes:
        age (int): The age of the customer.
        job (int): Numerical category for job type (0-3).
        housing (str): Housing status (e.g., 'own', 'rent').
        saving_accounts (str): Status of savings account.
        checking_account (str): Status of checking account.
        credit_amount (int): Total loan amount requested.
        duration (int): Loan duration in months.
        purpose (str): Stated reason for the loan.
        risk (str): The ML model's risk classification ('good' or 'bad').
    """
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(alias="Age")
    job: int = Field(alias="Job")
    housing: str = Field(alias="Housing")
    saving_accounts: Optional[str] = Field(default="unknown", alias="Saving accounts")
    checking_account: Optional[str] = Field(default="unknown", alias="Checking account")
    credit_amount: int = Field(alias="Credit amount")
    duration: int = Field(alias="Duration")
    purpose: str = Field(alias="Purpose")
    risk: str = Field(alias="Risk")


class RiskExplanation(BaseModel):
    """Schema for validating and structuring the LLM's natural language output."""
    description: str


class CreditLLMEngine:
    """
    Engine for generating human-readable explanations of credit risk predictions.
    
    This class orchestrates the communication between the application and the 
    LLM provider (Groq/OpenAI), handling prompt hydration and response validation.
    """

    def __init__(self, model: str = None, base_url: str = None, api_key: str = None):
        """
        Initializes the LLM client with tiered configuration resolution.
        
        Priority: 
        1. Explicit arguments passed during instantiation.
        2. Environment variables set in the OS/Cloud dashboard.
        3. Fallback defaults defined in the paths_config module.
        """
        self.model = model or os.getenv("MODEL_NAME") or paths_config.LLM_MODEL
        final_url = base_url or os.getenv("BASE_URL") or paths_config.BASE_URL
        final_key = api_key or os.getenv("API_KEY") or paths_config.API_KEY

        self.client = OpenAI(
            base_url=final_url,
            api_key=final_key
        )

    def _generate_prompt(self, data: CustomerData) -> str:
        """
        Loads the external prompt template and injects customer data.

        Args:
            data (CustomerData): The validated customer and prediction data.

        Returns:
            str: The fully hydrated prompt string ready for LLM inference.

        Raises:
            FileNotFoundError: If the template path in config is invalid.
        """
        try:
            template_path = paths_config.PROMPT_TEMPLATE_PATH
            
            with open(template_path, "r") as f:
                template_content = f.read()
            
            return template_content.format(
                risk=data.risk,
                credit_amount=data.credit_amount,
                duration=data.duration,
                purpose=data.purpose,
                checking_account=data.checking_account,
                saving_accounts=data.saving_accounts,
                age=data.age,
                job=data.job,
                housing=data.housing
            )
        except FileNotFoundError as e:
            print(f"ERROR: Template file missing: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Prompt generation failed: {e}")
            raise

    def get_description(self, data: CustomerData) -> str:
        """
        Submits data to the LLM and returns a natural language risk explanation.

        This method handles the high-level API call logic, system role definition,
        and provides a fallback message in case of service interruption.

        Args:
            data (CustomerData): The customer profile to explain.

        Returns:
            str: A short, simple explanation of the credit risk.
        """
        try:
            prompt = self._generate_prompt(data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional financial advisor who speaks simply. "
                                   "You never use technical jargon or bullet points."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.6 
            )
            
            content = response.choices[0].message.content
            result = RiskExplanation(description=content)
            return result.description

        except Exception as e:
            print(f"DEBUG: LLM Inference failed: {e}") 
            return "I'm sorry, I couldn't generate an explanation at this moment."
        
        
# # --- Usage Example (How to call this in your main app) ---
# if __name__ == "__main__":
#     # Initialize (will use env vars or defaults automatically)
#     engine = CreditLLMEngine()

#     # Example dictionary (mocking a row from your df)
#     example_row = {
#         "Age": 33, "Job": 2, "Housing": "own", 
#         "Saving accounts": "little", "Checking account": "moderate",
#         "Credit amount": 1169, "Duration": 6, "Purpose": "radio/TV", "Risk": "good"
#     }

#     # Create Pydantic model from dict
#     customer = CustomerData(**example_row)

#     # Get results
#     text = engine.get_description(customer)
#     print(text)
