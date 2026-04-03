import os
import re

print("Running fast patcher...")
target_dir = 'lib'

push_patterns = [
    (r"context\.push\('/home'\)", r"context.go('/home')"),
    (r"context\.push\('/cart'\)", r"context.go('/cart')"),
    (r"context\.push\('/orders'\)", r"context.go('/orders')"),
    (r"context\.push\('/profile'\)", r"context.go('/profile')"),
    (r"context\.push\('/search-tab'\)", r"context.go('/search-tab')"),
]

for root, _, files in os.walk(target_dir):
    for file in files:
        if not file.endswith('.dart'): continue
        filepath = os.path.join(root, file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            original_content = content
            
        for pat, rep in push_patterns:
            content = re.sub(pat, rep, content)
            
        # Also remove bottom navigation wrappers from the 4 primary screen files
        if file in ['home_screen.dart', 'cart_screen.dart', 'profile_screen.dart', 'orders_screen.dart']:
            # Replace bottomNavigationBar: _SomeNav(...)
            content = re.sub(r'\s*bottomNavigationBar:\s*[A-Za-z0-9_]+\([^)]*\),', '', content)
            # Replace bottomNavigationBar: Column/Container/SizedBox for any random ones not matching above
            content = re.sub(r'\s*bottomNavigationBar:.*?,\n(\s*)\);', r'\n\1);', content, flags=re.DOTALL)
            
        if file == 'splash_screen.dart':
             content = re.sub(r"context\.go\('/home'\);", r"context.go('/home');", content) # already go() by my earlier check, but just enforcing it.

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Patched: {filepath}")

print("Done")
