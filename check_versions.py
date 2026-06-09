from roboflow import Roboflow

try:
    print("Initializing Roboflow...")
    rf = Roboflow(api_key="APMubwY7a5CpLo2HJ8Yo")
    
    print("Loading workspace and project...")
    project = rf.workspace("mohamed-traore-2ekkp").project("chest-x-rays-qjmia")
    
    print("\nAvailable versions:")
    # Try to access the versions
    if hasattr(project, 'versions') and callable(project.versions):
        try:
            versions = project.versions()
            print(f"  Found {len(versions)} versions")
            for version_num in versions:
                print(f"  - Version {version_num}")
        except Exception as e:
            print(f"  Error retrieving versions: {e}")
    else:
        print("  Could not retrieve versions list")
    
    print("\n  Trying versions 1, 2, 3, 4, 5...")
    for v_num in [1, 2, 3, 4, 5]:
        try:
            version = project.version(v_num)
            print(f"  ✓ Version {v_num} exists")
        except Exception as e:
            print(f"  ✗ Version {v_num} error: {str(e)[:50]}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
