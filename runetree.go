package gpt_bpe

import "strings"

type RuneNode struct {
	rune      rune               // The rune this node represents.
	runes     []rune             // The prior runes that led to this node.
	terminal  bool               // If this node is an absolute terminal node.
	childs    map[rune]*RuneNode // The child nodes.
	childsArr *[]*RuneNode       // The child nodes in an array, for precedence
}

func runeIsIn(r rune, runes []rune) bool {
	for _, rr := range runes {
		if r == rr {
			return true
		}
	}
	return false
}

func (root *RuneNode) evaluate(node *RuneNode, r rune) (*RuneNode, bool) {
	// If the node has an array of children, use that. The array exists if the
	// node has less than 10 children, and is used to speed up the evaluation
	// of the node.
	if node.childsArr != nil {
		children := *node.childsArr
		for _, child := range children {
			if child.rune == r {
				return child, child.terminal
			}
		}
	} else {
		child, ok := node.childs[r]
		if ok {
			return child, child.terminal
		}
	}
	return nil, false
}

// Represent the tree as a string by traversing the tree, and using tree
// characters to represent the tree structure.
func (node *RuneNode) string(level int) string {
	if node == nil {
		return ""
	}
	s := string(node.rune)
	idx := 0
	if len(node.childs) == 1 {
		// Get the only element from the map recursively until we find a node
		// with more than one child.
		for r := range node.childs {
			s += node.childs[r].string(level)
		}
		return s
	}
	level += 1
	s += "\n"

	for r := range node.childs {
		childPrefix := strings.Repeat("| ", level-1)
		// If we're the last child, then we prepend with a tree terminator.
		if idx == len(node.childs)-1 {
			childPrefix += "└─"
		} else {
			childPrefix += "├─"
		}
		s += childPrefix + node.childs[r].string(level)
		idx += 1
	}
	return s
}

// Wrapper
func (node *RuneNode) String() string {
	return node.string(0)
}

func (encoder *GPTEncoder) createRuneTree() *RuneNode {
	runeTree := &RuneNode{
		runes:  []rune{},
		childs: make(map[rune]*RuneNode, 0),
	}

	for _, k := range encoder.specialsArr {
		keyRunes := []rune(k)
		keyLen := len(keyRunes)
		node := runeTree
		for i := 0; i < keyLen; i++ {
			r := keyRunes[i]
			childNode, ok := node.childs[r]
			if !ok {
				children := make([]*RuneNode, 0)
				node.childs[r] = &RuneNode{
					rune:      r,
					runes:     keyRunes[:i+1],
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
	}
	return runeTree
}
