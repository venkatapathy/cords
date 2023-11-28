from typing import List

class CharMatch:
    """
    Implements Character Error Rate (CER) metric for recognition tasks at character level.

    CER is computed as follows:

    CER = (Substitutions + Insertions + Deletions) / Total Number of Characters in Ground Truth
    """

    def __init__(self):
        self.reset()

    def update(self, gt: str, pred: str) -> None:
        """
        Update the state of the metric with new predictions

        Args:
        ----
            gt: ground-truth character sequence
            pred: predicted character sequence
        """
        subs, ins, dels = self.calculate_errors(gt, pred)
        self.substitutions += subs
        self.insertions += ins
        self.deletions += dels
        self.total_chars += len(gt)

    def calculate_errors(self, gt: str, pred: str) -> tuple:
        """ Calculate the number of substitutions, insertions, and deletions required to 
        transform pred into gt using dynamic programming (Levenshtein distance). """
        m, n = len(gt), len(pred)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif gt[i - 1] == pred[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],       # Insertion
                                       dp[i][j - 1],       # Deletion
                                       dp[i - 1][j - 1])   # Substitution

        # Extracting the number of insertions, deletions, and substitutions
        i, j = m, n
        substitutions, insertions, deletions = 0, 0, 0

        while i > 0 and j > 0:
            if gt[i - 1] == pred[j - 1]:
                i, j = i - 1, j - 1
            elif dp[i][j] == dp[i - 1][j - 1] + 1:
                substitutions += 1
                i, j = i - 1, j - 1
            elif dp[i][j] == dp[i - 1][j] + 1:
                deletions += 1
                i -= 1
            elif dp[i][j] == dp[i][j - 1] + 1:
                insertions += 1
                j -= 1

        return substitutions, insertions, deletions

    def summary(self) -> float:
        """
        Computes the aggregated CER

        Returns
        -------
            The Character Error Rate as a float
        """
        if self.total_chars == 0:
            raise AssertionError("You need to update the metric before getting the summary")

        cer = (self.substitutions + self.insertions + self.deletions) / self.total_chars
        return cer

    def compareLists(self,ground_truths: List[str], predictions: List[str]) -> float:
        """
        Computes the aggregated CER

        Returns
        -------
            The Character Error Rate as a float
        """
        if len(ground_truths) != len(predictions):
            raise AssertionError("You need to update the metric before getting the summary")

        char_match = CharMatch()

        # Iterating over each pair of ground truth and predicted text
        for gt, pred in zip(ground_truths, predictions):
            # Updating the CharMatch instance with the current pair
            char_match.update(gt, pred)

        # After all pairs have been processed, get the summary
        cer = char_match.summary()

        return cer

    def reset(self) -> None:
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.total_chars = 0

#run a main call
if __name__ == '__main__':
    from typing import List

    # Example usage of the CharMatch class
    def example_usage(ground_truths: List[str], predictions: List[str]) -> None:
        # Create an instance of the CharMatch class
        char_match = CharMatch()

        # Iterating over each pair of ground truth and predicted text
        for gt, pred in zip(ground_truths, predictions):
            # Updating the CharMatch instance with the current pair
            char_match.update(gt, pred)

        # After all pairs have been processed, get the summary
        cer = char_match.summary()

        print(f"Character Error Rate (CER): {cer * 100:.2f}%")
        #character recognition rate
        

    # Example ground truths and predictions
    ground_truths = ["hello world", "good morning", "how are you"]
    predictions = ["helo world", "god morning", "how ar you"]

    # Call the example usage function
    example_usage(ground_truths, predictions)