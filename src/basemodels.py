from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Any


class ParameterSpec(BaseModel):
    """Check and validate the function parameters"""
    model_config = ConfigDict(extra='forbid')
    type: str


class FunctionDefinition(BaseModel):
    """Check and validate the functions available"""
    model_config = ConfigDict(extra='ignore')
    name: str = Field(..., min_length=4)
    description: str = Field(...)
    parameters: dict[str, ParameterSpec] = Field(...)
    returns: ParameterSpec

    @model_validator(mode="before")
    @classmethod
    def check_extra_fields(cls, data: Any) -> Any:
        valid_fields = cls.model_fields.keys()

        extra_fields: set[str] = set(data) - set(valid_fields)

        for field in extra_fields:
            print(f"The field: '{field}' in function has been ignored")

        return data


class PromptItem(BaseModel):
    """Check and validate the test prompts"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def check_extra_fields(cls, data: Any) -> Any:
        valid_fields = cls.model_fields.keys()

        extra_fields: set[str] = set(data) - set(valid_fields)

        for field in extra_fields:
            print(f"The field: '{field}' in prompt has been ignored")

        return data


class FunctionCallResult(BaseModel):
    """Check and validate fuction call output"""
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(...)
    name: str = Field(..., min_length=4)
    parameters: dict[str, Any] = Field(...)
