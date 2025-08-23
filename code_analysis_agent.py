#!/usr/bin/env python3
"""
Intelligent Code Analysis Agent for Muse Super-Resolution Testing
Analyzes code and suggests necessary changes based on model configuration
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

class CodeAnalysisAgent:
    def __init__(self):
        self.issues_found = []
        self.suggestions = []
        self.model_config = {}
        
    def analyze_model_configuration(self, config_file: str) -> Dict:
        """Analyze the training configuration to understand model setup"""
        print("🔍 Analyzing model configuration...")
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Extract key configuration parameters
            config = {}
            
            # Check for DNA usage
            if 'self.use_dna = False' in content:
                config['use_dna'] = False
                print("  ✓ Model configured WITHOUT DNA context")
            elif 'self.use_dna = True' in content:
                config['use_dna'] = True
                print("  ✓ Model configured WITH DNA context")
            else:
                config['use_dna'] = None
                print("  ⚠ DNA usage not explicitly configured")
            
            # Check for DNA encoder initialization
            if 'self.dna_encoder = None' in content:
                config['has_dna_encoder'] = False
                print("  ✓ No DNA encoder initialized")
            else:
                config['has_dna_encoder'] = True
                print("  ✓ DNA encoder is initialized")
            
            # Check for super-resolution setup
            if 'cond_image_size' in content:
                config['has_super_resolution'] = True
                print("  ✓ Super-resolution model detected")
            else:
                config['has_super_resolution'] = False
                print("  ✓ Standard model (no super-resolution)")
            
            self.model_config = config
            return config
            
        except FileNotFoundError:
            print(f"❌ Configuration file {config_file} not found")
            return {}
    
    def analyze_test_file(self, test_file: str) -> List[Dict]:
        """Analyze the test file for potential issues"""
        print(f"\n🔍 Analyzing test file: {test_file}")
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            issues = []
            
            # Check for DNA coordinate usage
            dna_coords_patterns = [
                r'dna_coords=coords',
                r'dna_coords=coords_batch',
                r'dna_coords=\[coord\]',
                r'dna_coords=coords if self\.cfg\.use_dna else None'
            ]
            
            for pattern in dna_coords_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'type': 'dna_coords_usage',
                        'line': line_num,
                        'code': match.group(),
                        'severity': 'high' if self.model_config.get('use_dna') == False else 'medium',
                        'description': 'DNA coordinates being passed to model that may not use them'
                    })
            
            # Check for coordinate processing
            coord_processing_patterns = [
                r'coords = trainer\.ensure_coord_tuples\(coords\)',
                r'coords = \[tuple\(c\) for c in coords\.tolist\(\)\]'
            ]
            
            for pattern in coord_processing_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'type': 'coordinate_processing',
                        'line': line_num,
                        'code': match.group(),
                        'severity': 'medium' if self.model_config.get('use_dna') == False else 'low',
                        'description': 'Coordinate processing that may be unnecessary'
                    })
            
            # Check for generation method calls
            generation_patterns = [
                r'\.generate\([^)]*dna_coords[^)]*\)',
                r'\.generate\([^)]*texts=dummy_texts[^)]*\)'
            ]
            
            for pattern in generation_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'type': 'generation_call',
                        'line': line_num,
                        'code': match.group(),
                        'severity': 'high' if self.model_config.get('use_dna') == False else 'medium',
                        'description': 'Generation call that may need adjustment'
                    })
            
            return issues
            
        except FileNotFoundError:
            print(f"❌ Test file {test_file} not found")
            return []
    
    def generate_suggestions(self, issues: List[Dict]) -> List[Dict]:
        """Generate specific suggestions based on found issues"""
        print("\n💡 Generating suggestions...")
        
        suggestions = []
        
        for issue in issues:
            if issue['type'] == 'dna_coords_usage' and self.model_config.get('use_dna') == False:
                suggestions.append({
                    'type': 'remove_dna_coords',
                    'line': issue['line'],
                    'current_code': issue['code'],
                    'suggested_code': issue['code'].replace('dna_coords=coords', '').replace(', ,', ',').replace('(,', '(').replace(',)', ')'),
                    'reason': 'Model is configured without DNA context - remove dna_coords parameter',
                    'priority': 'high'
                })
            
            elif issue['type'] == 'coordinate_processing' and self.model_config.get('use_dna') == False:
                suggestions.append({
                    'type': 'remove_coord_processing',
                    'line': issue['line'],
                    'current_code': issue['code'],
                    'suggested_code': '# ' + issue['code'] + '  # Not needed for non-DNA model',
                    'reason': 'Coordinate processing not needed for model without DNA context',
                    'priority': 'medium'
                })
            
            elif issue['type'] == 'generation_call':
                # Suggest simplified generation calls
                current = issue['code']
                if 'dna_coords=coords' in current and self.model_config.get('use_dna') == False:
                    simplified = current.replace(', dna_coords=coords', '').replace('dna_coords=coords, ', '')
                    suggestions.append({
                        'type': 'simplify_generation',
                        'line': issue['line'],
                        'current_code': current,
                        'suggested_code': simplified,
                        'reason': 'Remove DNA coordinates from generation call',
                        'priority': 'high'
                    })
        
        return suggestions
    
    def create_fixed_code(self, test_file: str, suggestions: List[Dict]) -> str:
        """Create a fixed version of the code"""
        print("\n🔧 Creating fixed code...")
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Sort suggestions by line number (descending) to avoid line number shifts
            suggestions.sort(key=lambda x: x['line'], reverse=True)
            
            for suggestion in suggestions:
                line_idx = suggestion['line'] - 1  # Convert to 0-based index
                if line_idx < len(lines):
                    current_line = lines[line_idx]
                    
                    if suggestion['type'] == 'remove_dna_coords':
                        # Remove dna_coords parameter
                        new_line = current_line.replace('dna_coords=coords', '')
                        new_line = re.sub(r',\s*,', ',', new_line)  # Remove double commas
                        new_line = re.sub(r'\(\s*,', '(', new_line)  # Remove leading comma in parentheses
                        new_line = re.sub(r',\s*\)', ')', new_line)  # Remove trailing comma in parentheses
                        lines[line_idx] = new_line
                    
                    elif suggestion['type'] == 'remove_coord_processing':
                        # Comment out coordinate processing
                        lines[line_idx] = f"# {current_line}  # Not needed for non-DNA model"
                    
                    elif suggestion['type'] == 'simplify_generation':
                        # Replace the entire line
                        lines[line_idx] = current_line.replace(suggestion['current_code'], suggestion['suggested_code'])
            
            return '\n'.join(lines)
            
        except Exception as e:
            print(f"❌ Error creating fixed code: {e}")
            return ""
    
    def generate_comprehensive_report(self, issues: List[Dict], suggestions: List[Dict]) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("INTELLIGENT CODE ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Model Configuration Summary
        report.append("\n📋 MODEL CONFIGURATION SUMMARY:")
        report.append("-" * 40)
        for key, value in self.model_config.items():
            status = "✓" if value else "✗" if value is False else "?"
            report.append(f"  {status} {key}: {value}")
        
        # Issues Found
        report.append(f"\n🚨 ISSUES FOUND ({len(issues)}):")
        report.append("-" * 40)
        for issue in issues:
            severity_icon = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
            report.append(f"  {severity_icon} Line {issue['line']}: {issue['description']}")
            report.append(f"     Code: {issue['code']}")
        
        # Suggestions
        report.append(f"\n💡 SUGGESTIONS ({len(suggestions)}):")
        report.append("-" * 40)
        for suggestion in suggestions:
            priority_icon = "🔴" if suggestion['priority'] == 'high' else "🟡" if suggestion['priority'] == 'medium' else "🟢"
            report.append(f"  {priority_icon} Line {suggestion['line']}: {suggestion['reason']}")
            report.append(f"     Current: {suggestion['current_code']}")
            report.append(f"     Suggested: {suggestion['suggested_code']}")
            report.append("")
        
        # Summary
        high_priority = len([s for s in suggestions if s['priority'] == 'high'])
        report.append(f"\n📊 SUMMARY:")
        report.append("-" * 40)
        report.append(f"  Total issues found: {len(issues)}")
        report.append(f"  Total suggestions: {len(suggestions)}")
        report.append(f"  High priority fixes: {high_priority}")
        
        if self.model_config.get('use_dna') == False:
            report.append("\n🎯 KEY RECOMMENDATION:")
            report.append("  Your model is configured WITHOUT DNA context.")
            report.append("  Remove all 'dna_coords' parameters from generation calls.")
            report.append("  Keep 'cond_images' parameters for super-resolution functionality.")
        
        return '\n'.join(report)
    
    def run_analysis(self, config_file: str, test_file: str, output_file: str = None):
        """Run the complete analysis"""
        print("🤖 Starting Intelligent Code Analysis Agent...")
        
        # Analyze model configuration
        self.analyze_model_configuration(config_file)
        
        # Analyze test file
        issues = self.analyze_test_file(test_file)
        
        # Generate suggestions
        suggestions = self.generate_suggestions(issues)
        
        # Generate report
        report = self.generate_comprehensive_report(issues, suggestions)
        
        # Print report
        print(report)
        
        # Create fixed code if requested
        if output_file:
            fixed_code = self.create_fixed_code(test_file, suggestions)
            if fixed_code:
                with open(output_file, 'w') as f:
                    f.write(fixed_code)
                print(f"\n✅ Fixed code saved to: {output_file}")
        
        return report, suggestions

def main():
    parser = argparse.ArgumentParser(description='Intelligent Code Analysis Agent for Muse Testing')
    parser.add_argument('--config', type=str, default='muse_pipeline_improved.py',
                       help='Training configuration file')
    parser.add_argument('--test-file', type=str, default='test_superresolution.py',
                       help='Test file to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for fixed code (optional)')
    
    args = parser.parse_args()
    
    agent = CodeAnalysisAgent()
    report, suggestions = agent.run_analysis(args.config, args.test_file, args.output)
    
    print(f"\n🎉 Analysis complete! Found {len(suggestions)} suggestions.")

if __name__ == "__main__":
    main()


