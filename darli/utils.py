class RecursiveNamespace:
    @staticmethod
    def map_entry(entry: dict) -> "RecursiveNamespace":
        """
        Static method that maps the given dictionary `entry` to an instance of RecursiveNamespace class.

        Args:
        - entry (dict): The dictionary to be mapped.

        Returns:
        - RecursiveNamespace: The instance of RecursiveNamespace class.
        """
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        """
        Initializes the RecursiveNamespace instance with the given keyword arguments.

        Args:
        - **kwargs: The keyword arguments to be set as attributes of the instance.
        """
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)
