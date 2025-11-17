"""
Generic Data Analysis Tools for CSV files
Can analyze any CSV file and provide insights, quality checks, and improvement suggestions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ListCSVsTool(BaseTool):
    """Tool to list all available CSV files."""

    name: str = "list_csv_files"
    description: str = """
    Lists all CSV files available in the data directory.
    Use this to discover what data files are available for analysis.
    No input needed - just call it to see all CSV files.
    """

    data_dir: str = Field(default="datos")

    def __init__(self, data_dir: str = "datos"):
        super().__init__(data_dir=data_dir)

    def _run(self, query: str = "") -> str:
        """List all CSV files."""
        data_path = Path(self.data_dir)

        if not data_path.exists():
            return f"Error: Directory {self.data_dir} does not exist"

        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            return f"No CSV files found in {self.data_dir}"

        result = f"\nFOUND {len(csv_files)} CSV FILES:\n"
        result += "="*80 + "\n\n"

        for i, file in enumerate(csv_files, 1):
            size_mb = file.stat().st_size / (1024 * 1024)
            result += f"{i}. {file.name} ({size_mb:.2f} MB)\n"

        result += "\n" + "="*80
        result += "\n\nUse 'inspect_csv' tool to analyze a specific file."

        return result

    async def _arun(self, query: str = "") -> str:
        """Async version."""
        return self._run(query)


class InspectCSVTool(BaseTool):
    """Tool to inspect the structure of a CSV file."""

    name: str = "inspect_csv_structure"
    description: str = """
    Inspects the structure and basic information of a CSV file.
    Input should be the filename (e.g., 'orders.csv' or 'olist_orders_dataset.csv')
    Returns: columns, data types, shape, sample rows, memory usage.
    """

    data_dir: str = Field(default="datos")

    def __init__(self, data_dir: str = "datos"):
        super().__init__(data_dir=data_dir)

    def _run(self, filename: str) -> str:
        """Inspect CSV structure."""
        try:
            file_path = Path(self.data_dir) / filename

            if not file_path.exists():
                return f"Error: File {filename} not found in {self.data_dir}"

            # Read CSV with minimal rows for structure inspection
            df = pd.read_csv(file_path, nrows=5)
            full_df = pd.read_csv(file_path)

            result = f"\nFILE: {filename}\n"
            result += "="*80 + "\n\n"

            # Basic info
            result += f"SHAPE: {full_df.shape[0]:,} rows x {full_df.shape[1]} columns\n"
            result += f"MEMORY: {full_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n\n"

            # Column information
            result += "COLUMNS:\n"
            result += "-"*80 + "\n"

            for col in df.columns:
                dtype = full_df[col].dtype
                non_null = full_df[col].notna().sum()
                null_pct = (full_df[col].isna().sum() / len(full_df)) * 100

                result += f"\n{col}:\n"
                result += f"  Type: {dtype}\n"
                result += f"  Non-null: {non_null:,} ({100-null_pct:.1f}%)\n"

                if null_pct > 0:
                    result += f"  Missing: {full_df[col].isna().sum():,} ({null_pct:.1f}%)\n"

                # Sample values
                if dtype == 'object':
                    unique_count = full_df[col].nunique()
                    result += f"  Unique values: {unique_count:,}\n"
                    if unique_count <= 10:
                        result += f"  Values: {full_df[col].unique()[:10].tolist()}\n"
                elif np.issubdtype(dtype, np.number):
                    result += f"  Range: {full_df[col].min()} to {full_df[col].max()}\n"
                    result += f"  Mean: {full_df[col].mean():.2f}\n"

            # Sample rows
            result += "\n" + "-"*80
            result += "\nSAMPLE ROWS (first 3):\n"
            result += "-"*80 + "\n"
            result += full_df.head(3).to_string(index=False)

            result += "\n\n" + "="*80

            return result

        except Exception as e:
            return f"Error inspecting {filename}: {str(e)}"

    async def _arun(self, filename: str) -> str:
        """Async version."""
        return self._run(filename)


class AnalyzeCSVTool(BaseTool):
    """Tool for comprehensive data analysis."""

    name: str = "analyze_csv_data"
    description: str = """
    Performs comprehensive exploratory data analysis (EDA) on a CSV file.
    Input should be the filename.
    Returns: statistical summary, distributions, correlations, patterns.
    """

    data_dir: str = Field(default="datos")

    def __init__(self, data_dir: str = "datos"):
        super().__init__(data_dir=data_dir)

    def _run(self, filename: str) -> str:
        """Perform EDA."""
        try:
            file_path = Path(self.data_dir) / filename

            if not file_path.exists():
                return f"Error: File {filename} not found"

            df = pd.read_csv(file_path)

            result = f"\nCOMPREHENSIVE ANALYSIS: {filename}\n"
            result += "="*80 + "\n\n"

            # 1. Dataset Overview
            result += "DATASET OVERVIEW:\n"
            result += "-"*80 + "\n"
            result += f"Rows: {len(df):,}\n"
            result += f"Columns: {len(df.columns)}\n"
            result += f"Total cells: {df.size:,}\n"
            result += f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n\n"

            # 2. Data Types
            result += "DATA TYPES:\n"
            result += "-"*80 + "\n"
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                result += f"  {dtype}: {count} columns\n"
            result += "\n"

            # 3. Missing Values
            result += "MISSING VALUES:\n"
            result += "-"*80 + "\n"
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            has_missing = missing[missing > 0].sort_values(ascending=False)

            if len(has_missing) > 0:
                result += f"Columns with missing values: {len(has_missing)}\n\n"
                for col, count in has_missing.items():
                    result += f"  {col}: {count:,} ({missing_pct[col]:.2f}%)\n"
            else:
                result += "No missing values found!\n"
            result += "\n"

            # 4. Numerical Columns Analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result += "NUMERICAL COLUMNS STATISTICS:\n"
                result += "-"*80 + "\n"
                stats = df[numeric_cols].describe()
                result += stats.to_string() + "\n\n"

                # Check for outliers (IQR method)
                result += "OUTLIERS DETECTED (IQR method):\n"
                result += "-"*80 + "\n"
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                    if len(outliers) > 0:
                        pct = len(outliers) / len(df) * 100
                        result += f"  {col}: {len(outliers):,} outliers ({pct:.2f}%)\n"
                result += "\n"

            # 5. Categorical Columns Analysis
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                result += "CATEGORICAL COLUMNS:\n"
                result += "-"*80 + "\n"
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    result += f"\n{col}:\n"
                    result += f"  Unique values: {unique_count:,}\n"

                    if unique_count <= 20:
                        top_values = df[col].value_counts().head(10)
                        result += f"  Top values:\n"
                        for val, count in top_values.items():
                            pct = count / len(df) * 100
                            result += f"    {val}: {count:,} ({pct:.1f}%)\n"
                    else:
                        result += f"  (High cardinality - {unique_count} unique values)\n"
                result += "\n"

            # 6. Correlations (for numeric columns)
            if len(numeric_cols) > 1:
                result += "CORRELATIONS (top 10 strongest):\n"
                result += "-"*80 + "\n"
                corr_matrix = df[numeric_cols].corr()

                # Get correlation pairs
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        correlations.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })

                # Sort by absolute correlation
                correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

                for corr in correlations[:10]:
                    result += f"  {corr['col1']} <-> {corr['col2']}: {corr['correlation']:.3f}\n"
                result += "\n"

            # 7. Duplicates
            result += "DUPLICATE ROWS:\n"
            result += "-"*80 + "\n"
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                result += f"Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)\n"
            else:
                result += "No duplicate rows found.\n"
            result += "\n"

            result += "="*80

            return result

        except Exception as e:
            return f"Error analyzing {filename}: {str(e)}"

    async def _arun(self, filename: str) -> str:
        """Async version."""
        return self._run(filename)


class DataQualityTool(BaseTool):
    """Tool to check data quality issues."""

    name: str = "check_data_quality"
    description: str = """
    Checks for data quality issues in a CSV file.
    Input should be the filename.
    Returns: quality score, issues found, severity of problems.
    """

    data_dir: str = Field(default="datos")

    def __init__(self, data_dir: str = "datos"):
        super().__init__(data_dir=data_dir)

    def _run(self, filename: str) -> str:
        """Check data quality."""
        try:
            file_path = Path(self.data_dir) / filename

            if not file_path.exists():
                return f"Error: File {filename} not found"

            df = pd.read_csv(file_path)

            result = f"\nDATA QUALITY REPORT: {filename}\n"
            result += "="*80 + "\n\n"

            issues = []
            score = 100.0

            # 1. Missing Values Check
            missing_pct = (df.isnull().sum().sum() / df.size) * 100
            if missing_pct > 0:
                severity = "HIGH" if missing_pct > 20 else "MEDIUM" if missing_pct > 5 else "LOW"
                issues.append({
                    'issue': 'Missing Values',
                    'severity': severity,
                    'description': f'{missing_pct:.2f}% of data is missing',
                    'impact': -min(missing_pct, 30)
                })
                score -= min(missing_pct, 30)

            # 2. Duplicate Rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                dup_pct = duplicates / len(df) * 100
                severity = "HIGH" if dup_pct > 10 else "MEDIUM" if dup_pct > 1 else "LOW"
                issues.append({
                    'issue': 'Duplicate Rows',
                    'severity': severity,
                    'description': f'{duplicates:,} duplicate rows ({dup_pct:.2f}%)',
                    'impact': -min(dup_pct * 2, 20)
                })
                score -= min(dup_pct * 2, 20)

            # 3. Data Type Issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if should be numeric
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        issues.append({
                            'issue': 'Incorrect Data Type',
                            'severity': 'MEDIUM',
                            'description': f'{col} stored as text but contains numbers',
                            'impact': -5
                        })
                        score -= 5
                    except:
                        pass

                    # Check if should be datetime
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            pd.to_datetime(df[col], errors='raise')
                            issues.append({
                                'issue': 'Incorrect Data Type',
                                'severity': 'MEDIUM',
                                'description': f'{col} should be datetime but is text',
                                'impact': -5
                            })
                            score -= 5
                        except:
                            pass

            # 4. Outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            total_outliers = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                total_outliers += len(outliers)

            if total_outliers > 0:
                outlier_pct = total_outliers / df.size * 100
                severity = "HIGH" if outlier_pct > 5 else "MEDIUM" if outlier_pct > 1 else "LOW"
                issues.append({
                    'issue': 'Outliers Detected',
                    'severity': severity,
                    'description': f'{total_outliers:,} outlier values ({outlier_pct:.2f}%)',
                    'impact': -min(outlier_pct * 2, 15)
                })
                score -= min(outlier_pct * 2, 15)

            # 5. High Cardinality
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5 and df[col].nunique() > 100:
                    issues.append({
                        'issue': 'High Cardinality',
                        'severity': 'LOW',
                        'description': f'{col} has {df[col].nunique():,} unique values ({unique_ratio*100:.1f}% of rows)',
                        'impact': -5
                    })
                    score -= 5

            # 6. Zero Variance Columns
            for col in numeric_cols:
                if df[col].nunique() == 1:
                    issues.append({
                        'issue': 'Zero Variance',
                        'severity': 'MEDIUM',
                        'description': f'{col} has only one unique value',
                        'impact': -10
                    })
                    score -= 10

            # Ensure score doesn't go below 0
            score = max(score, 0)

            # Generate report
            result += f"OVERALL QUALITY SCORE: {score:.1f}/100\n"

            if score >= 90:
                result += "Rating: EXCELLENT\n"
            elif score >= 75:
                result += "Rating: GOOD\n"
            elif score >= 60:
                result += "Rating: FAIR\n"
            elif score >= 40:
                result += "Rating: POOR\n"
            else:
                result += "Rating: CRITICAL\n"

            result += "\n" + "-"*80 + "\n"

            if issues:
                result += f"\nISSUES FOUND: {len(issues)}\n\n"

                # Group by severity
                high = [i for i in issues if i['severity'] == 'HIGH']
                medium = [i for i in issues if i['severity'] == 'MEDIUM']
                low = [i for i in issues if i['severity'] == 'LOW']

                if high:
                    result += "HIGH SEVERITY:\n"
                    for issue in high:
                        result += f"  [!] {issue['issue']}: {issue['description']}\n"
                    result += "\n"

                if medium:
                    result += "MEDIUM SEVERITY:\n"
                    for issue in medium:
                        result += f"  [-] {issue['issue']}: {issue['description']}\n"
                    result += "\n"

                if low:
                    result += "LOW SEVERITY:\n"
                    for issue in low:
                        result += f"  [~] {issue['issue']}: {issue['description']}\n"
                    result += "\n"

            else:
                result += "\nNo quality issues found!\n"

            result += "="*80
            result += "\n\nUse 'suggest_improvements' tool for recommendations on fixing these issues."

            return result

        except Exception as e:
            return f"Error checking quality of {filename}: {str(e)}"

    async def _arun(self, filename: str) -> str:
        """Async version."""
        return self._run(filename)


class DataImprovementTool(BaseTool):
    """Tool to suggest data improvements."""

    name: str = "suggest_data_improvements"
    description: str = """
    Suggests specific improvements and transformations for a CSV file.
    Input should be the filename.
    Returns: actionable recommendations with code examples.
    """

    data_dir: str = Field(default="datos")

    def __init__(self, data_dir: str = "datos"):
        super().__init__(data_dir=data_dir)

    def _run(self, filename: str) -> str:
        """Suggest improvements."""
        try:
            file_path = Path(self.data_dir) / filename

            if not file_path.exists():
                return f"Error: File {filename} not found"

            df = pd.read_csv(file_path)

            result = f"\nDATA IMPROVEMENT RECOMMENDATIONS: {filename}\n"
            result += "="*80 + "\n\n"

            recommendations = []

            # 1. Missing Values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                result += "1. HANDLE MISSING VALUES:\n"
                result += "-"*80 + "\n"
                for col, count in missing[missing > 0].items():
                    pct = count / len(df) * 100
                    result += f"\n{col} - {count:,} missing ({pct:.1f}%):\n"

                    if pct > 50:
                        result += "  Recommendation: Consider dropping this column\n"
                        result += f"  Code: df.drop('{col}', axis=1)\n"
                    elif df[col].dtype == 'object':
                        result += "  Recommendation: Fill with 'Unknown' or most frequent\n"
                        result += f"  Code: df['{col}'].fillna('Unknown', inplace=True)\n"
                    else:
                        result += "  Recommendation: Fill with median or mean\n"
                        result += f"  Code: df['{col}'].fillna(df['{col}'].median(), inplace=True)\n"
                result += "\n"

            # 2. Data Type Conversions
            result += "2. DATA TYPE CONVERSIONS:\n"
            result += "-"*80 + "\n"
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check for dates
                    if 'date' in col.lower() or 'time' in col.lower():
                        result += f"\n{col}:\n"
                        result += "  Recommendation: Convert to datetime\n"
                        result += f"  Code: df['{col}'] = pd.to_datetime(df['{col}'])\n"

                    # Check for categories
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.05:  # Less than 5% unique
                        result += f"\n{col}:\n"
                        result += "  Recommendation: Convert to category (saves memory)\n"
                        result += f"  Code: df['{col}'] = df['{col}'].astype('category')\n"
            result += "\n"

            # 3. Handle Outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_cols = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    outlier_cols.append(col)

            if outlier_cols:
                result += "3. HANDLE OUTLIERS:\n"
                result += "-"*80 + "\n"
                for col in outlier_cols:
                    result += f"\n{col}:\n"
                    result += "  Option 1: Cap outliers (winsorization)\n"
                    result += f"  Code:\n"
                    result += f"    Q1 = df['{col}'].quantile(0.25)\n"
                    result += f"    Q3 = df['{col}'].quantile(0.75)\n"
                    result += f"    IQR = Q3 - Q1\n"
                    result += f"    df['{col}'] = df['{col}'].clip(Q1-1.5*IQR, Q3+1.5*IQR)\n\n"
                    result += "  Option 2: Remove outliers\n"
                    result += f"  Code: df = df[(df['{col}'] >= Q1-1.5*IQR) & (df['{col}'] <= Q3+1.5*IQR)]\n"
                result += "\n"

            # 4. Remove Duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                result += "4. REMOVE DUPLICATE ROWS:\n"
                result += "-"*80 + "\n"
                result += f"Found {duplicates:,} duplicate rows\n"
                result += "Code: df.drop_duplicates(inplace=True)\n\n"

            # 5. Feature Engineering Suggestions
            result += "5. FEATURE ENGINEERING IDEAS:\n"
            result += "-"*80 + "\n"

            # Date features
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                result += "\nFrom datetime columns, extract:\n"
                for col in date_cols:
                    result += f"  - Year, month, day, hour from {col}\n"
                    result += f"  - Day of week, is_weekend from {col}\n"
                result += "\n"

            # Categorical encoding
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                result += "\nCategorical encoding:\n"
                for col in cat_cols:
                    unique = df[col].nunique()
                    if unique <= 10:
                        result += f"  - One-hot encode {col} ({unique} categories)\n"
                        result += f"    Code: pd.get_dummies(df['{col}'], prefix='{col}')\n"
                    else:
                        result += f"  - Label encode {col} ({unique} categories - high cardinality)\n"
                        result += f"    Code: from sklearn.preprocessing import LabelEncoder\n"
                        result += f"          le = LabelEncoder()\n"
                        result += f"          df['{col}_encoded'] = le.fit_transform(df['{col}'])\n"
                result += "\n"

            # 6. Scaling
            if len(numeric_cols) > 0:
                result += "6. SCALING/NORMALIZATION:\n"
                result += "-"*80 + "\n"
                result += "For machine learning, consider scaling numerical features:\n\n"
                result += "Option 1: StandardScaler (z-score normalization)\n"
                result += "Code:\n"
                result += "  from sklearn.preprocessing import StandardScaler\n"
                result += "  scaler = StandardScaler()\n"
                result += "  df[numeric_cols] = scaler.fit_transform(df[numeric_cols])\n\n"
                result += "Option 2: MinMaxScaler (0-1 range)\n"
                result += "Code:\n"
                result += "  from sklearn.preprocessing import MinMaxScaler\n"
                result += "  scaler = MinMaxScaler()\n"
                result += "  df[numeric_cols] = scaler.fit_transform(df[numeric_cols])\n\n"

            result += "="*80
            result += "\n\nIMPLEMENT THESE IMPROVEMENTS TO ENHANCE DATA QUALITY!"

            return result

        except Exception as e:
            return f"Error generating improvements for {filename}: {str(e)}"

    async def _arun(self, filename: str) -> str:
        """Async version."""
        return self._run(filename)


def get_generic_data_tools(data_dir: str = "datos") -> List[BaseTool]:
    """Get all generic data analysis tools."""
    return [
        ListCSVsTool(data_dir=data_dir),
        InspectCSVTool(data_dir=data_dir),
        AnalyzeCSVTool(data_dir=data_dir),
        DataQualityTool(data_dir=data_dir),
        DataImprovementTool(data_dir=data_dir)
    ]
