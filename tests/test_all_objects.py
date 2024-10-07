# Navigate to working directory in command line
# Run:   test_dev_module.py
# Great video for help: https://www.bing.com/videos/riverview/relatedvideo?q=python+what+are+unit+tests&mid=559EF026DB1A3B82BD0E559EF026DB1A3B82BD0E&FORM=VIRE
# Useful unit testing docs: https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug

import unittest
import dev_module


class TestDevModule(unittest.TestCase):
    def test_get_nis(self):
        flu_2023 = dev_module.get_nis(
            data_path="https://data.cdc.gov/api/views/2v3t-r3np/rows.csv?accessType=DOWNLOAD\u0026bom=true\u0026format=true",
            region_col="Geographic_Name",
            date_col="Current_Season_Week_Ending",
            estimate_col="ND_Weekly_Estimate",
            filters={
                "Geographic_Name": "National",
                "Demographic_Level": "Overall",
                "Demographic_Name": "18+ years",
                "Indicator_Category_Label": "Received a vaccination",
            },
        )
        self.assertIsInstance(flu_2023, dev_module.CumulativeUptakeData)


if __name__ == "__main__":
    unittest.main()
