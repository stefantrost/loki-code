package main

// toolEntry binds a tool name to its executor and its plan-mode policy.
// Adding a new tool here lets ExecuteToolWithPlanMode dispatch to it without
// editing any switch statements.
type toolEntry struct {
	exec        func(args map[string]interface{}) (string, error)
	planAllowed bool
}

// toolRegistry maps tool name → entry. Order doesn't matter; lookup is by name.
var toolRegistry = map[string]toolEntry{
	"create_file":  {exec: executeCreateFile, planAllowed: false},
	"read_file":    {exec: executeReadFile, planAllowed: true},
	"update_file":  {exec: executeUpdateFile, planAllowed: false},
	"delete_file":  {exec: executeDeleteFile, planAllowed: false},
	"list_files":   {exec: executeListFiles, planAllowed: true},
	"exec_command": {exec: executeCommand, planAllowed: false},
	"find_files":   {exec: executeFindFiles, planAllowed: true},
	"grep_content": {exec: executeGrepContent, planAllowed: true},
	"get_pwd":      {exec: executeGetPwd, planAllowed: true},
	"tree_view":    {exec: executeTreeView, planAllowed: true},
	"analyze_code": {exec: executeAnalyzeCode, planAllowed: true},
	"http_request": {exec: executeCurl, planAllowed: true},
}
