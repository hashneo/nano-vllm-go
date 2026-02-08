package nanovllm

import (
	"encoding/binary"

	"github.com/cespare/xxhash/v2"
)

// Block represents a KV cache block
type Block struct {
	BlockID  int
	RefCount int
	Hash     uint64
	TokenIDs []int
}

// NewBlock creates a new block
func NewBlock(blockID int) *Block {
	return &Block{
		BlockID:  blockID,
		RefCount: 0,
		Hash:     0,
		TokenIDs: make([]int, 0),
	}
}

// Update updates the block's hash and token IDs
func (b *Block) Update(hash uint64, tokenIDs []int) {
	b.Hash = hash
	b.TokenIDs = make([]int, len(tokenIDs))
	copy(b.TokenIDs, tokenIDs)
}

// Reset resets the block for reuse
func (b *Block) Reset() {
	b.RefCount = 1
	b.Hash = 0
	b.TokenIDs = make([]int, 0)
}

// BlockManager manages KV cache blocks with prefix caching
type BlockManager struct {
	blockSize     int
	blocks        []*Block
	hashToBlockID map[uint64]int
	freeBlockIDs  []int
	usedBlockIDs  map[int]bool
}

// NewBlockManager creates a new block manager
func NewBlockManager(numBlocks int, blockSize int) *BlockManager {
	blocks := make([]*Block, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blocks[i] = NewBlock(i)
	}

	freeBlockIDs := make([]int, numBlocks)
	for i := 0; i < numBlocks; i++ {
		freeBlockIDs[i] = i
	}

	return &BlockManager{
		blockSize:     blockSize,
		blocks:        blocks,
		hashToBlockID: make(map[uint64]int),
		freeBlockIDs:  freeBlockIDs,
		usedBlockIDs:  make(map[int]bool),
	}
}

// ComputeHash computes the hash of token IDs with an optional prefix hash
func (bm *BlockManager) ComputeHash(tokenIDs []int, prefixHash uint64) uint64 {
	h := xxhash.New()

	if prefixHash != 0 {
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, prefixHash)
		h.Write(buf)
	}

	for _, tokenID := range tokenIDs {
		buf := make([]byte, 4)
		binary.LittleEndian.PutUint32(buf, uint32(tokenID))
		h.Write(buf)
	}

	return h.Sum64()
}

// allocateBlock allocates a block
func (bm *BlockManager) allocateBlock(blockID int) *Block {
	block := bm.blocks[blockID]
	if block.RefCount != 0 {
		panic("block is already allocated")
	}

	block.Reset()

	// Remove from free list
	for i, id := range bm.freeBlockIDs {
		if id == blockID {
			bm.freeBlockIDs = append(bm.freeBlockIDs[:i], bm.freeBlockIDs[i+1:]...)
			break
		}
	}

	bm.usedBlockIDs[blockID] = true
	return block
}

// deallocateBlock deallocates a block
func (bm *BlockManager) deallocateBlock(blockID int) {
	block := bm.blocks[blockID]
	if block.RefCount != 0 {
		panic("block still has references")
	}

	delete(bm.usedBlockIDs, blockID)
	bm.freeBlockIDs = append(bm.freeBlockIDs, blockID)
}

// CanAllocate checks if there are enough free blocks for a sequence
func (bm *BlockManager) CanAllocate(seq *Sequence) bool {
	return len(bm.freeBlockIDs) >= seq.NumBlocks()
}

// Allocate allocates blocks for a sequence with prefix caching
func (bm *BlockManager) Allocate(seq *Sequence) {
	if len(seq.BlockTable) > 0 {
		panic("sequence already has blocks allocated")
	}

	var h uint64 = 0
	cacheMiss := false

	for i := 0; i < seq.NumBlocks(); i++ {
		tokenIDs := seq.Block(i)

		// Compute hash only for full blocks
		if len(tokenIDs) == bm.blockSize {
			h = bm.ComputeHash(tokenIDs, h)
		} else {
			h = 0
		}

		blockID := -1
		if h != 0 {
			if id, ok := bm.hashToBlockID[h]; ok {
				blockID = id
			}
		}

		// Check if cached block matches
		if blockID != -1 && bm.blocks[blockID].TokenIDs != nil {
			match := true
			if len(bm.blocks[blockID].TokenIDs) != len(tokenIDs) {
				match = false
			} else {
				for j, tid := range tokenIDs {
					if bm.blocks[blockID].TokenIDs[j] != tid {
						match = false
						break
					}
				}
			}
			if !match {
				blockID = -1
			}
		} else {
			blockID = -1
		}

		if blockID == -1 {
			cacheMiss = true
		}

		if cacheMiss {
			// Allocate new block
			blockID = bm.freeBlockIDs[0]
			block := bm.allocateBlock(blockID)
			_ = block
		} else {
			// Use cached block
			seq.NumCachedTokens += bm.blockSize
			if bm.usedBlockIDs[blockID] {
				// Block is already in use, increment ref count
				block := bm.blocks[blockID]
				block.RefCount++
			} else {
				// Block is free but cached, allocate it
				bm.allocateBlock(blockID)
			}
		}

		// Update block metadata
		if h != 0 {
			bm.blocks[blockID].Update(h, tokenIDs)
			bm.hashToBlockID[h] = blockID
		}

		seq.BlockTable = append(seq.BlockTable, blockID)
	}
}

// Deallocate deallocates blocks for a sequence
func (bm *BlockManager) Deallocate(seq *Sequence) {
	// Deallocate in reverse order
	for i := len(seq.BlockTable) - 1; i >= 0; i-- {
		blockID := seq.BlockTable[i]
		block := bm.blocks[blockID]
		block.RefCount--
		if block.RefCount == 0 {
			bm.deallocateBlock(blockID)
		}
	}

	seq.NumCachedTokens = 0
	seq.BlockTable = seq.BlockTable[:0]
}

// CanAppend checks if a new token can be appended to a sequence
func (bm *BlockManager) CanAppend(seq *Sequence) bool {
	needsNewBlock := seq.Len()%bm.blockSize == 1
	if needsNewBlock {
		return len(bm.freeBlockIDs) >= 1
	}
	return true
}

// MayAppend prepares for appending a token to a sequence
func (bm *BlockManager) MayAppend(seq *Sequence) {
	blockTable := seq.BlockTable
	lastBlockIdx := len(blockTable) - 1
	lastBlock := bm.blocks[blockTable[lastBlockIdx]]

	if seq.Len()%bm.blockSize == 1 {
		// Need to allocate a new block
		if lastBlock.Hash == 0 {
			panic("last block should have a hash")
		}
		blockID := bm.freeBlockIDs[0]
		bm.allocateBlock(blockID)
		seq.BlockTable = append(seq.BlockTable, blockID)
	} else if seq.Len()%bm.blockSize == 0 {
		// Block is now full, compute hash
		if lastBlock.Hash != 0 {
			panic("last block should not have a hash")
		}
		tokenIDs := seq.Block(seq.NumBlocks() - 1)
		var prefixHash uint64 = 0
		if len(blockTable) > 1 {
			prefixHash = bm.blocks[blockTable[lastBlockIdx-1]].Hash
		}
		h := bm.ComputeHash(tokenIDs, prefixHash)
		lastBlock.Update(h, tokenIDs)
		bm.hashToBlockID[h] = lastBlock.BlockID
	} else {
		// Still filling the block
		if lastBlock.Hash != 0 {
			panic("last block should not have a hash")
		}
	}
}
