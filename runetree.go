package gpt_bpe

import "strings"

type RuneNode struct {
	rune        rune               // The rune this node represents.
	runes       []rune             // The prior runes that led to this node.
	terminal    bool               // If this node is an absolute terminal node.
	replacement *[]rune            // The replacement runes for this node.
	childs      map[rune]*RuneNode // The child nodes.
	childsArr   *[]*RuneNode       // The child nodes in an array, for precedence
}

type RuneNodes []*RuneNode

func runeIsIn(r rune, runes []rune) bool {
	for _, rr := range runes {
		if r == rr {
			return true
		}
	}
	return false
}

func (nodes *RuneNodes) evaluate(r rune) *RuneNode {
	var idx int
	var candidate *RuneNode
	for idx, candidate = range *nodes {
		candidate = candidate.evaluate(r)
		(*nodes)[idx] = candidate
		if candidate != nil && (candidate.terminal || candidate.
			replacement != nil) {
			break
		}
	}
	// Clean out any nodes that are no longer valid.
	for idx = 0; idx < len(*nodes); idx++ {
		if idx >= len(*nodes) {
			break
		}
		if (*nodes)[idx] == nil {
			*nodes = append((*nodes)[:idx], (*nodes)[idx+1:]...)
			idx--
		}
	}
	return candidate
}

func (node *RuneNode) evaluate(r rune) *RuneNode {
	// If the node has an array of children, use that. The array exists if the
	// node has less than 10 children, and is used to speed up the evaluation
	// of the node.
	if node.childsArr != nil {
		children := *node.childsArr
		for _, child := range children {
			if child.rune == r {
				return child
			}
		}
	} else {
		child, ok := node.childs[r]
		if ok {
			return child
		}
	}
	return nil
}

// Represent the tree as a string by traversing the tree, and using tree
// characters to represent the tree structure.
func (node *RuneNode) string(level int, sb *strings.Builder) {
	if node == nil {
		return
	}
	sb.WriteRune(node.rune)
	idx := 0
	if len(node.childs) == 1 {
		// Get the only element from the map recursively until we find a node
		// with more than one child.
		for r := range node.childs {
			node.childs[r].string(level, sb)
		}
		return
	}
	level += 1
	if node.replacement != nil {
		sb.WriteString(" -> ")
		sb.WriteString(string(*node.replacement))
	}
	sb.WriteByte('\n')

	for r := range node.childs {
		sb.WriteString(strings.Repeat("| ", level-1))
		// If we're the last child, then we prepend with a tree terminator.
		if idx == len(node.childs)-1 {
			sb.WriteString("└─")
		} else {
			sb.WriteString("├─")
		}
		node.childs[r].string(level, sb)
		idx += 1
	}
}

// Wrapper
func (runeTree *RuneNode) String() string {
	sb := strings.Builder{}
	runeTree.string(0, &sb)
	return sb.String()
}

func (runeTree *RuneNode) insertRunes(runes []rune) (node *RuneNode) {
	node = runeTree
	keyLen := len(runes)
	for i := 0; i < keyLen; i++ {
		r := runes[i]
		childNode, ok := node.childs[r]
		if !ok {
			children := make([]*RuneNode, 0)
			node.childs[r] = &RuneNode{
				rune:      r,
				runes:     runes[:i+1],
				terminal:  i == keyLen-1,
				childs:    make(map[rune]*RuneNode, 0),
				childsArr: &children,
			}
		} else if i == keyLen-1 {
			childNode.terminal = true
		}
		if len(node.childs) > 10 {
			// If there are more than 10 children, we set the array pointer
			// to nil, so that we can use the map instead.
			node.childsArr = nil
		} else {
			if node.childsArr == nil {
				children := make([]*RuneNode, 0)
				node.childsArr = &children
			}
			if len(node.childs) != len(*node.childsArr) {
				*node.childsArr = append(*node.childsArr, node.childs[r])
			}
		}
		node = node.childs[r]
	}
	return node
}

func NewRuneTree() *RuneNode {
	return &RuneNode{
		runes:  []rune{},
		childs: make(map[rune]*RuneNode, 0),
	}
}

func (runeTree *RuneNode) InsertReplacementsIntoRuneTree(
	replacements map[string]string,
) {
	for k, v := range replacements {
		keyRunes := []rune(k)
		valueRunes := []rune(v)
		keyNode := runeTree.insertRunes(keyRunes)
		keyNode.replacement = &valueRunes
	}
}

func CreateReplacementsRuneTree(replacements map[string]string) *RuneNode {
	runeTree := NewRuneTree()
	runeTree.InsertReplacementsIntoRuneTree(replacements)
	return runeTree
}

func (runeTree *RuneNode) InsertIntoRuneTree(s []string) {
	for _, k := range s {
		keyRunes := []rune(k)
		runeTree.insertRunes(keyRunes)
	}
}

// Create a new rune tree from an array of strings to match against.
func CreateRuneTree(s []string) *RuneNode {
	runeTree := NewRuneTree()
	runeTree.InsertIntoRuneTree(s)
	return runeTree
}
