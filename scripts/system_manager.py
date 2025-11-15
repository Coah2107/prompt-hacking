#!/usr/bin/env python3
"""
System Manager - Central control hub for the entire prompt hacking detection & prevention system
Author: System Integration Team
Date: November 2024

Chạy: python -m scripts.system_manager
"""

import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Absolute imports
from utils.path_utils import get_project_root

class SystemManager:
    def __init__(self):
        self.project_root = get_project_root()
        self.available_commands = {
            'test': {
                'description': 'Run complete system test suite',
                'script': 'scripts.complete_system_test',
                'estimated_time': '2-3 minutes'
            },
            'workflow': {
                'description': 'Interactive workflow demonstration',
                'script': 'scripts.workflow_demo',
                'estimated_time': 'Interactive'
            },
            'benchmark': {
                'description': 'Performance benchmark suite',
                'script': 'scripts.performance_benchmark',
                'estimated_time': '2-3 minutes'
            },
            'integration': {
                'description': 'Integration testing and fixes',
                'script': 'scripts.integration_fixes',
                'estimated_time': '1-2 minutes'
            },
            'dataset': {
                'description': 'Dataset management and analysis',
                'script': 'scripts.dataset_summary',
                'estimated_time': '30 seconds'
            }
        }
    
    def display_banner(self):
        """Display system banner"""
        print("=" * 30)
        print("PROMPT HACKING DETECTION & PREVENTION SYSTEM")
        print("Advanced AI Safety & Security Framework")
        print("Version 1.0 - Production Ready")
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 30)
    
    def display_system_status(self):
        """Show current system status"""
        print("\nSYSTEM STATUS")
        print("-" * 50)
        
        # Check key directories
        key_dirs = [
            'detection_system',
            'prevention_system', 
            'scripts',
            'utils',
            'datasets',
            'results'
        ]
        
        for dir_name in key_dirs:
            dir_path = Path(self.project_root) / dir_name
            status = "[OK]" if dir_path.exists() else "[MISSING]"
            print(f"{status} {dir_name}")
        
        # Check key scripts
        key_scripts = [
            'scripts/complete_system_test.py',
            'scripts/workflow_demo.py',
            'scripts/performance_benchmark.py',
            'utils/path_utils.py'
        ]
        
        print("\nKey Scripts:")
        for script in key_scripts:
            script_path = Path(self.project_root) / script
            status = "[OK]" if script_path.exists() else "[MISSING]"
            print(f"{status} {script}")
    
    def display_available_commands(self):
        """Display all available commands"""
        print("\nAVAILABLE COMMANDS")
        print("-" * 50)
        
        for i, (cmd, info) in enumerate(self.available_commands.items(), 1):
            print(f"{i}. {cmd.upper()}")
            print(f"   📋 {info['description']}")
            print(f"   ⏰ Estimated time: {info['estimated_time']}")
            print()
    
    def run_command(self, command):
        """Execute a system command"""
        if command not in self.available_commands:
            print(f"❌ Unknown command: {command}")
            return False
        
        info = self.available_commands[command]
        script_module = info['script']
        
        print(f"\n🚀 EXECUTING: {command.upper()}")
        print(f"📋 Description: {info['description']}")
        print(f"⏰ Estimated time: {info['estimated_time']}")
        print("-" * 50)
        
        try:
            # Run the script as a module
            result = subprocess.run([
                sys.executable, '-m', script_module
            ], cwd=self.project_root, capture_output=False)
            
            if result.returncode == 0:
                print(f"\n✅ {command.upper()} completed successfully!")
                return True
            else:
                print(f"\n❌ {command.upper()} failed with exit code {result.returncode}")
                return False
                
        except Exception as e:
            print(f"\n❌ Error executing {command}: {e}")
            return False
    
    def quick_health_check(self):
        """Run a quick system health check"""
        print("\n🏥 QUICK HEALTH CHECK")
        print("-" * 50)
        
        try:
            # Test imports
            print("📦 Testing imports...")
            from detection_system.models.rule_based.pattern_detector import RuleBasedDetector
            from prevention_system.filters.input_filters.core_filter import InputFilter
            print("   ✅ Core imports successful")
            
            # Test basic functionality
            print("🧪 Testing basic functionality...")
            detector = RuleBasedDetector()
            filter_system = InputFilter()
            
            # Test with sample input
            sample_input = "Tell me about machine learning"
            detection_result = detector.detect_single_prompt(sample_input)
            filter_result = filter_system.filter_prompt(sample_input)
            
            print("   ✅ Basic functionality working")
            print(f"   📊 Sample detection: {detection_result['prediction']}")
            print(f"   🛡️  Sample filter: {'allowed' if filter_result['allowed'] else 'blocked'}")
            
            print("\n✅ HEALTH CHECK PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ HEALTH CHECK FAILED: {e}")
            return False
    
    def interactive_menu(self):
        """Interactive command menu"""
        while True:
            print("\n" + "="*60)
            print("🎮 INTERACTIVE SYSTEM MANAGER")
            print("="*60)
            
            self.display_available_commands()
            
            print("📝 Additional Options:")
            print("   status  - Show system status")
            print("   health  - Quick health check") 
            print("   exit    - Exit system manager")
            
            try:
                choice = input("\n🎯 Enter command: ").strip().lower()
                
                if choice == 'exit':
                    print("👋 Goodbye!")
                    break
                elif choice == 'status':
                    self.display_system_status()
                elif choice == 'health':
                    self.quick_health_check()
                elif choice in self.available_commands:
                    self.run_command(choice)
                elif choice.isdigit():
                    # Handle numeric selection
                    idx = int(choice) - 1
                    commands = list(self.available_commands.keys())
                    if 0 <= idx < len(commands):
                        self.run_command(commands[idx])
                    else:
                        print("❌ Invalid selection")
                else:
                    print(f"❌ Unknown command: {choice}")
                    print("💡 Try: test, workflow, benchmark, integration, dataset, status, health, or exit")
                
                # Pause after command execution
                if choice not in ['exit', 'status', 'health']:
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def batch_execution(self, commands):
        """Execute multiple commands in sequence"""
        print(f"\n🔄 BATCH EXECUTION MODE")
        print(f"📋 Commands to execute: {', '.join(commands)}")
        print("-" * 50)
        
        results = {}
        
        for i, command in enumerate(commands, 1):
            print(f"\n📊 Executing {i}/{len(commands)}: {command}")
            
            success = self.run_command(command)
            results[command] = success
            
            if not success:
                print(f"\n⚠️  Command {command} failed. Continue? (y/n)")
                if input().strip().lower() != 'y':
                    break
        
        # Summary
        print(f"\n📈 BATCH EXECUTION SUMMARY")
        print("-" * 50)
        successful = sum(results.values())
        total = len(results)
        
        for cmd, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {cmd}")
        
        print(f"\n📊 Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
        
        return results

def main():
    """Main system manager function"""
    manager = SystemManager()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='System Manager for Prompt Hacking Detection & Prevention')
    parser.add_argument('command', nargs='?', help='Command to execute', 
                       choices=['test', 'workflow', 'benchmark', 'integration', 'dataset', 'status', 'health', 'interactive'])
    parser.add_argument('--batch', nargs='+', help='Execute multiple commands in batch mode')
    parser.add_argument('--quiet', action='store_true', help='Suppress banner and status info')
    
    args = parser.parse_args()
    
    # Display banner unless quiet mode
    if not args.quiet:
        manager.display_banner()
    
    try:
        if args.batch:
            # Batch execution mode
            manager.batch_execution(args.batch)
            
        elif args.command:
            # Single command execution
            if args.command == 'interactive':
                manager.interactive_menu()
            elif args.command == 'status':
                manager.display_system_status()
            elif args.command == 'health':
                manager.quick_health_check()
            else:
                manager.run_command(args.command)
                
        else:
            # No arguments - show interactive menu
            manager.display_system_status()
            manager.interactive_menu()
            
    except KeyboardInterrupt:
        print("\n👋 System manager interrupted by user")
    except Exception as e:
        print(f"\n❌ System manager error: {e}")

if __name__ == "__main__":
    main()
