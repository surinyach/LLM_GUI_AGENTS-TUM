def parse_llm_response(response_text: str) -> str:
        """
        Parses the text response from an LLM, expecting a "RESPONSE:" prefix.

        This helper function extracts the relevant content from the LLM's response
        by splitting it at the "RESPONSE:" delimiter and stripping any leading/trailing
        whitespace. It raises a ValueError if the expected prefix is not found.

        Args:
            response_text (str): The raw text response received from the LLM.

        Returns:
            str: The extracted content after the "RESPONSE:" prefix.
        """
        parts = response_text.split("RESPONSE:", 1)
        if len(parts) < 2:
            raise ValueError(f"LLM response missing 'RESPONSE:' prefix: {response_text}")
        
        return parts[1].strip()

