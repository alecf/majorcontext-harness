package engine

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestClaudeCodeAskUserQuestionGate pins the reported problem: without
// --permission-prompt-tool stdio the child never registers AskUserQuestion.
// The flag must stay off unless the embedder opts in, and off for a task
// child or a goal-supervised session, where nothing answers a parked call.
func TestClaudeCodeAskUserQuestionGate(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ask    bool
		parent string
		goal   bool
		want   bool
	}{
		{"dormant by default", false, "", false, false},
		{"root session opted in", true, "", false, true},
		{"task child", true, "ses_parent", false, false},
		{"goal armed", true, "", true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, logPath := claudeCodeTestSession(t, "normal")
			s.cfg.ClaudeCode.AskUserQuestion = tc.ask
			s.cfg.TaskParentID = tc.parent
			if tc.goal {
				if err := s.RegisterGoal("ship it"); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := s.Prompt(context.Background(), "hi"); err != nil {
				t.Fatalf("Prompt: %v", err)
			}
			argv := readInvocations(t, logPath)[0]
			if got, _ := argvValueAfter(argv, "--permission-prompt-tool"); (got == "stdio") != tc.want {
				t.Fatalf("--permission-prompt-tool stdio present = %v, want %v: %v", got == "stdio", tc.want, argv)
			}
			settings, _ := argvValueAfter(argv, "--settings")
			disallowed, _ := argvValueAfter(argv, "--disallowedTools")
			if tc.want && (!strings.Contains(settings, `"matcher":"AskUserQuestion"`) || !strings.Contains(settings, "defer") ||
				!strings.HasSuffix(disallowed, ",EnterPlanMode,ExitPlanMode")) {
				t.Errorf("opted-in argv lacks the defer hook or the plan-mode ban: settings=%q disallowed=%q", settings, disallowed)
			}
		})
	}
}

// TestClaudeCodeUnansweredPermissionRequestIsDenied pins the hang the stdio
// permission surface introduces: outside bypassPermissions an ordinary tool
// sends can_use_tool and blocks until the host answers it on stdin.
func TestClaudeCodeUnansweredPermissionRequestIsDenied(t *testing.T) {
	s, _ := claudeCodeTestSession(t, "normal")
	frames := `{"type":"control_request","request_id":"r1","request":{"subtype":"can_use_tool","tool_name":"Bash","tool_use_id":"toolu_b","input":{}}}` + "\n" +
		`{"type":"result","subtype":"success","num_turns":1}` + "\n"
	var sent []string
	s.consumeClaudeCodeStream(strings.NewReader(frames), s.Model(), func(b []byte) { sent = append(sent, string(b)) })
	if len(sent) != 1 {
		t.Fatalf("control responses sent = %d, want 1: %q", len(sent), sent)
	}
	got := decodeControlResponse(t, sent[0])
	if got.Response.RequestID != "r1" || got.Response.Response.Behavior != "deny" {
		t.Errorf("control response = %s, want a deny for r1", sent[0])
	}
}

// TestClaudeCodeAnswerResumesParkedQuestion pins the park-and-resume
// contract: a tool_deferred result is a parked turn, not a finished one, and
// the answer turn writes no driving text, only the answer for that call.
func TestClaudeCodeAnswerResumesParkedQuestion(t *testing.T) {
	s, stdinLog := claudeCodeQuestionSession(t)
	if _, err := s.Prompt(context.Background(), "pick a db"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := s.PendingQuestion(); got != "toolu_q" {
		t.Fatalf("PendingQuestion() after the parked turn = %q, want toolu_q", got)
	}
	if _, err := s.AnswerQuestion(context.Background(), "toolu_q", map[string]string{"Which database?": "SQLite"}); err != nil {
		t.Fatalf("AnswerQuestion: %v", err)
	}
	lines := readStdinLines(t, stdinLog)
	if len(lines) != 2 {
		t.Fatalf("stdin lines = %d, want the prompt then one control response: %q", len(lines), lines)
	}
	got := decodeControlResponse(t, lines[1])
	if got.Response.Response.Behavior != "allow" || got.Response.Response.UpdatedInput.Answers["Which database?"] != "SQLite" ||
		len(got.Response.Response.UpdatedInput.Questions) != 1 {
		t.Errorf("answer turn control response = %s, want allow with the questions and the answer", lines[1])
	}
	if s.PendingQuestion() != "" {
		t.Errorf("PendingQuestion() after the answer = %q, want empty", s.PendingQuestion())
	}
}

// TestClaudeCodePromptDismissesParkedQuestion pins the lost-reply hazard: a
// resumed CLI re-runs its deferred call before it reads stdin, so a prompt
// written on that turn is answered in a second result harness never reads.
// The parked call must be dismissed first, with no text on that child.
func TestClaudeCodePromptDismissesParkedQuestion(t *testing.T) {
	s, stdinLog := claudeCodeQuestionSession(t)
	if _, err := s.Prompt(context.Background(), "pick a db"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	msg, err := s.Prompt(context.Background(), "use the default")
	if err != nil {
		t.Fatalf("second Prompt: %v", err)
	}
	lines := readStdinLines(t, stdinLog)
	if len(lines) != 3 || !strings.Contains(lines[2], "use the default") {
		t.Fatalf("stdin lines = %q, want prompt, dismissal, then the new prompt", lines)
	}
	got := decodeControlResponse(t, lines[1])
	if got.Response.Response.Behavior != "deny" || !got.Response.Response.Interrupt {
		t.Errorf("dismissal = %s, want deny with interrupt", lines[1])
	}
	if s.PendingQuestion() != "" || msg.Parts.Text() != "Done — it printed hi." {
		t.Errorf("after dismissal PendingQuestion()=%q reply=%q, want empty and the new turn's reply", s.PendingQuestion(), msg.Parts.Text())
	}
}

func claudeCodeQuestionSession(t *testing.T) (*Session, string) {
	t.Helper()
	s, _ := claudeCodeTestSession(t, "question")
	s.cfg.ClaudeCode.AskUserQuestion = true
	dir := t.TempDir()
	t.Setenv("FAKE_CLAUDE_STATE", filepath.Join(dir, "parked"))
	stdinLog := filepath.Join(dir, "stdin.log")
	t.Setenv("FAKE_CLAUDE_STDIN_LOG", stdinLog)
	return s, stdinLog
}

func readStdinLines(t *testing.T, path string) []string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimRight(string(data), "\n"), "\n")
}

type controlResponseForTest struct {
	Response struct {
		RequestID string `json:"request_id"`
		Response  struct {
			Behavior     string `json:"behavior"`
			Interrupt    bool   `json:"interrupt"`
			UpdatedInput struct {
				Questions []json.RawMessage `json:"questions"`
				Answers   map[string]string `json:"answers"`
			} `json:"updatedInput"`
		} `json:"response"`
	} `json:"response"`
}

func decodeControlResponse(t *testing.T, line string) controlResponseForTest {
	t.Helper()
	var c controlResponseForTest
	if err := json.Unmarshal([]byte(line), &c); err != nil {
		t.Fatalf("decoding control response %q: %v", line, err)
	}
	return c
}
