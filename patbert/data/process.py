
class BaseProcessor():
    def __init__(self) -> None:
        pass

    def group_rare_values(self, df, col, threshold=0.01):
        counts = df[col].value_counts(normalize=True)
        rare_values = counts[counts < threshold].index
        df.loc[df[col].isin(rare_values), col] = 'OTHER'