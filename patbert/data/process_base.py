
class BaseProcessor():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def group_rare_values(self, df, col, category='OTHER'):
        """Rare values, below the threshold, are grouped into a single category,"""
        counts = df[col].value_counts(normalize=False)
        rare_values = counts[counts < self.cfg.patients_info.rare_threshold].index
        df.loc[df[col].isin(rare_values), col] = category
        return df