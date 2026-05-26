package tools

import (
	"os"
	"path/filepath"
	"testing"
)

// TestValidatePath_NestedTraversal covers traversal sequences where ".." sits
// after a valid-looking prefix. The pre-fix implementation rejected only
// literal "..", so paths like "a/../../etc/passwd" sneaked through when "a"
// resolved within the cwd.
func TestValidatePath_NestedTraversal(t *testing.T) {
	cases := []string{
		"a/../../etc/passwd",
		"src/sub/../../../outside",
		"./valid/../../../escape",
	}
	for _, p := range cases {
		t.Run(p, func(t *testing.T) {
			if err := validatePath(p); err == nil {
				t.Errorf("validatePath(%q) should have errored", p)
			}
		})
	}
}

// TestValidatePath_AbsoluteCWD ensures absolute paths within the cwd are accepted.
func TestValidatePath_AbsoluteCWD(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	abs := filepath.Join(cwd, "some", "file.txt")
	if err := validatePath(abs); err != nil {
		t.Errorf("expected absolute path inside cwd to be accepted, got %v", err)
	}
}

// TestValidateURL_DockerBridge verifies the new CIDR-based blocker rejects
// addresses in 172.16.0.0/12, including the Docker default bridge range.
func TestValidateURL_DockerBridge(t *testing.T) {
	urls := []string{
		"http://172.17.0.1/api",
		"http://172.20.5.5/",
		"http://[fc00::1]/",              // IPv6 ULA
		"http://[fe80::1]/",              // IPv6 link-local
		"http://169.254.169.254/latest/meta-data/", // AWS metadata
	}
	for _, u := range urls {
		t.Run(u, func(t *testing.T) {
			if err := validateURL(u); err == nil {
				t.Errorf("validateURL(%q) should have errored", u)
			}
		})
	}
}

// TestValidateURL_PublicAllowed sanity-checks that ordinary public hostnames
// still pass. We don't actually hit the network; LookupIP failures are
// permissive by design.
func TestValidateURL_PublicAllowed(t *testing.T) {
	if err := validateURL("https://example.com/path"); err != nil {
		t.Errorf("expected example.com to pass, got %v", err)
	}
}
