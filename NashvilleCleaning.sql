-- Cleaning Data in SQL Queries

*/

Select *
From [PortfolioProject(3)].dbo.NashvilleHousing

-----------------------------------------------------------------------------------------

-- Standardize Date Format

Select SaleDateConverted, CONVERT(Date, SaleDate)
From [PortfolioProject(3)].dbo.NashvilleHousing

--Command not updating
Update [PortfolioProject(3)].dbo.NashvilleHousing
SET SaleDate = CONVERT(Date, SaleDate)

-- Create new column instead
ALTER TABLE NashvilleHousing
Add SaleDateConverted Date;

Update [PortfolioProject(3)].dbo.NashvilleHousing
SET SaleDateConverted = CONVERT(Date, SaleDate)

-----------------------------------------------------------------------------------------

--Populate Property Address data

Select PropertyAddress
From [PortfolioProject(3)].dbo.NashvilleHousing
--Where PropertyAddress is null
order by ParcelID

Select a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress, b.PropertyAddress)
From [PortfolioProject(3)].dbo.NashvilleHousing a
Join [PortfolioProject(3)].dbo.NashvilleHousing b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
Where a.PropertyAddress is null


Update a
SET PropertyAddress = ISNULL(a.PropertyAddress, b.PropertyAddress)
From [PortfolioProject(3)].dbo.NashvilleHousing a
Join [PortfolioProject(3)].dbo.NashvilleHousing b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
Where a.PropertyAddress is null

-----------------------------------------------------------------------------------------

--Breaking out Property Address into Individual Columns (Address, City, State)

Select PropertyAddress
From [PortfolioProject(3)].dbo.NashvilleHousing
order by ParcelID

SELECT
SUBSTRING(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) -1) as Address,
--Starts at first char ends before comma
SUBSTRING(PropertyAddress, CHARINDEX(',', PropertyAddress) +1, LEN(PropertyAddress)) as Address
--starts after comma ends at last char
From [PortfolioProject(3)].dbo.NashvilleHousing

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
Add StreetAddress Nvarchar(255);

Update [PortfolioProject(3)].dbo.NashvilleHousing
SET StreetAddress = SUBSTRING(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) -1)

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
Add City Nvarchar(255);

Update [PortfolioProject(3)].dbo.NashvilleHousing
SET City = SUBSTRING(PropertyAddress, CHARINDEX(',', PropertyAddress) +1, LEN(PropertyAddress))

Select *
From [PortfolioProject(3)].dbo.NashvilleHousing

USE [PortfolioProject(3)]
GO
EXEC sp_rename 'NashvilleHousing.City', 'PropertyCity', 'COLUMN';
GO

USE [PortfolioProject(3)]
GO
EXEC sp_rename 'NashvilleHousing.StreetAddress', 'PropertyStreet', 'COLUMN';
GO

-----------------------------------------------------------------------------------------

--Breaking out Owner Address into Individual Columns (Address, City, State)

SELECT OwnerAddress 
From [PortfolioProject(3)].dbo.NashvilleHousing

SELECT 
PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3)
,PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2)
,PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1)
From [PortfolioProject(3)].dbo.NashvilleHousing

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
Add OwnerStreet Nvarchar(255);

Update [PortfolioProject(3)].dbo.NashvilleHousing
SET OwnerStreet = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3)

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
Add OwnerCity Nvarchar(255);

Update [PortfolioProject(3)].dbo.NashvilleHousing
SET OwnerCity = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2)

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
Add OwnerState Nvarchar(255);

Update [PortfolioProject(3)].dbo.NashvilleHousing
SET OwnerState = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1)

Select *
From [PortfolioProject(3)].dbo.NashvilleHousing

---------------------------------------------------------------------------------------------

--Change Y and N in "Sold as Vacant"

Select Distinct(SoldASVacant), Count(SoldASVacant)
From [PortfolioProject(3)].dbo.NashvilleHousing
Group by SoldAsVacant
order by 2

Select SoldAsVacant
, CASE When SoldAsVacant = 'Y' THEN 'Yes'
	When SoldASVacant = 'N' THEN 'No'
	ELSE SoldAsVacant
	END
From [PortfolioProject(3)].dbo.NashvilleHousing

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
Add SoldASVacant Nvarchar(255);

Update [PortfolioProject(3)].dbo.NashvilleHousing
SET SoldAsVacant = 
 CASE When SoldAsVacant = 'Y' THEN 'Yes'
	When SoldASVacant = 'N' THEN 'No'
	ELSE SoldAsVacant
	END

----------------------------------------------------------------------------------------

--Remove Duplicates
--Create Temp Table to Query out dupliates (with matching values in partition columns)

WITH RowNumCTE AS(
Select *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID, 
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER BY
					UniqueID
					)row_num

From [PortfolioProject(3)].dbo.NashvilleHousing
--order by ParcelID
)
Select *
--DELETE
From RowNumCTE
Where row_num > 1
Order by PropertyAddress

---------------------------------------------------------------------------------------

--Delete Unused Columns

Select *
From [PortfolioProject(3)].dbo.NashvilleHousing

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
DROP COLUMN OwnerAddress, TaxDistrict, PropertyAddress

ALTER TABLE [PortfolioProject(3)].dbo.NashvilleHousing
DROP COLUMN SaleDate